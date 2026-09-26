package turnlog

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Emitter writes turn telemetry as day-partitioned JSONL. It is designed around one rule: it may
// lose data, but it may never slow down or fail a turn. Every enqueue is non-blocking and a full
// queue drops the record and bumps a counter that the nightly probe reads — a silently dropped
// event is a known, counted loss, whereas a blocked agent is a user-visible regression.
//
// A nil *Emitter is a valid no-op receiver for every method. Wire points inside crush therefore
// need no nil checks and no build tags: when telemetry is disabled, New returns nil and the calls
// compile away to a nil-receiver branch.
type Emitter struct {
	cfg Config

	queue   chan any
	done    chan struct{}
	wg      sync.WaitGroup
	closeMu sync.Mutex
	closed  bool

	// writeMu guards the file handle. Turn-close records bypass the queue and write directly, so
	// the writer goroutine and the closing caller can contend here.
	writeMu sync.Mutex
	file    *os.File
	fileDay string
	pending int

	dropped   atomic.Int64
	written   atomic.Int64
	writeErrs atomic.Int64
}

type Config struct {
	Dir           string
	Host          string
	Source        string
	QueueSize     int
	RetentionDays int
	// TextCapture is "full", "truncated" or "off". Prompts and responses are the most useful and
	// most dangerous thing captured; this is the knob that decides how much of them lands on disk.
	TextCapture string
	MaxTextLen  int
	// SyncEvery forces an fsync after this many buffered records. Turn-close records always fsync
	// regardless: losing an event costs detail, losing a turn record costs the whole row.
	SyncEvery int
}

const (
	TextCaptureFull      = "full"
	TextCaptureTruncated = "truncated"
	TextCaptureOff       = "off"
)

func DefaultConfig() Config {
	return Config{
		Dir:           defaultDir(),
		Host:          defaultHost(),
		Source:        SourceCrush,
		QueueSize:     4096,
		RetentionDays: 14,
		TextCapture:   TextCaptureFull,
		MaxTextLen:    64 * 1024,
		SyncEvery:     32,
	}
}

// ConfigFromEnv layers environment overrides onto DefaultConfig. Env rather than crush.json for the
// kill switch specifically, so telemetry can be turned off for a single invocation without editing
// shared config that another box has symlinked.
func ConfigFromEnv() (Config, bool) {
	cfg := DefaultConfig()
	if v := os.Getenv("SPARKY_TURNLOG"); v == "0" || strings.EqualFold(v, "false") || strings.EqualFold(v, "off") {
		return cfg, false
	}
	if v := os.Getenv("SPARKY_TURNLOG_DIR"); v != "" {
		cfg.Dir = expandHome(v)
	}
	if v := os.Getenv("SPARKY_HOST_LABEL"); v != "" {
		cfg.Host = v
	}
	if v := os.Getenv("SPARKY_TURNLOG_SOURCE"); v != "" {
		cfg.Source = v
	}
	if v := os.Getenv("SPARKY_TURNLOG_TEXT"); v != "" {
		cfg.TextCapture = strings.ToLower(v)
	}
	if v := os.Getenv("SPARKY_TURNLOG_QUEUE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.QueueSize = n
		}
	}
	return cfg, true
}

// New returns an Emitter, or nil if telemetry is disabled or the directory is unusable. Returning
// nil rather than an error is deliberate: no caller inside crush should have to decide what to do
// when telemetry cannot start, and none of them should abort a turn over it.
func New(cfg Config) *Emitter {
	if cfg.Dir == "" {
		return nil
	}
	if err := os.MkdirAll(cfg.Dir, 0o700); err != nil {
		return nil
	}
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 4096
	}
	if cfg.MaxTextLen <= 0 {
		cfg.MaxTextLen = 64 * 1024
	}
	if cfg.SyncEvery <= 0 {
		cfg.SyncEvery = 32
	}
	e := &Emitter{
		cfg:   cfg,
		queue: make(chan any, cfg.QueueSize),
		done:  make(chan struct{}),
	}
	e.wg.Add(1)
	go e.loop()
	return e
}

// NewFromEnv is the constructor the crush wire points call.
func NewFromEnv() *Emitter {
	cfg, enabled := ConfigFromEnv()
	if !enabled {
		return nil
	}
	return New(cfg)
}

func (e *Emitter) loop() {
	defer e.wg.Done()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case rec := <-e.queue:
			e.write(rec, false)
		case <-ticker.C:
			e.sync()
		case <-e.done:
			// Drain whatever is already queued, then stop. Anything still being produced after
			// Close is dropped on purpose — shutdown must be bounded.
			for {
				select {
				case rec := <-e.queue:
					e.write(rec, false)
				default:
					e.sync()
					return
				}
			}
		}
	}
}

func (e *Emitter) enqueue(rec any) {
	if e == nil {
		return
	}
	select {
	case e.queue <- rec:
	default:
		e.dropped.Add(1)
	}
}

func (e *Emitter) write(rec any, forceSync bool) {
	line, err := json.Marshal(rec)
	if err != nil {
		e.writeErrs.Add(1)
		return
	}
	e.writeMu.Lock()
	defer e.writeMu.Unlock()
	if err := e.ensureFileLocked(); err != nil {
		e.writeErrs.Add(1)
		return
	}
	if _, err := e.file.Write(append(line, '\n')); err != nil {
		e.writeErrs.Add(1)
		return
	}
	e.written.Add(1)
	e.pending++
	if forceSync || e.pending >= e.cfg.SyncEvery {
		_ = e.file.Sync()
		e.pending = 0
	}
}

func (e *Emitter) sync() {
	e.writeMu.Lock()
	defer e.writeMu.Unlock()
	if e.file != nil && e.pending > 0 {
		_ = e.file.Sync()
		e.pending = 0
	}
}

func (e *Emitter) ensureFileLocked() error {
	day := time.Now().UTC().Format("2006-01-02")
	if e.file != nil && e.fileDay == day {
		return nil
	}
	if e.file != nil {
		_ = e.file.Sync()
		_ = e.file.Close()
		e.file = nil
	}
	path := filepath.Join(e.cfg.Dir, "events-"+day+".jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	e.file, e.fileDay, e.pending = f, day, 0
	return nil
}

// Close flushes the queue and closes the file. Safe to call more than once.
func (e *Emitter) Close() {
	if e == nil {
		return
	}
	e.closeMu.Lock()
	if e.closed {
		e.closeMu.Unlock()
		return
	}
	e.closed = true
	close(e.done)
	e.closeMu.Unlock()

	e.wg.Wait()

	// Report our own losses before shutting down. A drop that nobody records is exactly the
	// silent-failure mode this project exists to eliminate.
	stats := e.Stats()
	total := stats.Written + stats.Dropped + stats.WriteErrs
	detail := encodePayload(map[string]any{
		"written": stats.Written, "dropped": stats.Dropped, "write_errors": stats.WriteErrs,
		"source": e.cfg.Source, "queue_size": e.cfg.QueueSize,
	})
	e.write(probeRecord{
		Kind:   kindProbe,
		TS:     nowMS(),
		Host:   e.cfg.Host,
		Probe:  "collector_heartbeat",
		OK:     int(stats.Written),
		Total:  int(total),
		Detail: detail,
	}, true)

	e.writeMu.Lock()
	defer e.writeMu.Unlock()
	if e.file != nil {
		_ = e.file.Sync()
		_ = e.file.Close()
		e.file = nil
	}
}

// Stats reports what the emitter did and, more usefully, what it lost. The nightly probe treats a
// nonzero drop count as a finding: it means the queue was too small for the workload, and the
// KPIs computed from that period are understated.
type Stats struct {
	Written   int64
	Dropped   int64
	WriteErrs int64
	Queued    int
}

func (e *Emitter) Stats() Stats {
	if e == nil {
		return Stats{}
	}
	return Stats{
		Written:   e.written.Load(),
		Dropped:   e.dropped.Load(),
		WriteErrs: e.writeErrs.Load(),
		Queued:    len(e.queue),
	}
}

func (e *Emitter) retentionMS() int64 {
	days := e.cfg.RetentionDays
	if days <= 0 {
		days = 14
	}
	return int64(days) * 24 * 60 * 60 * 1000
}

func defaultDir() string {
	if v := os.Getenv("SPARKY_TURNLOG_DIR"); v != "" {
		return expandHome(v)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".local", "state", "sparky", "turnlog")
}

func defaultHost() string {
	if v := os.Getenv("SPARKY_HOST_LABEL"); v != "" {
		return v
	}
	name, err := os.Hostname()
	if err != nil {
		return "unknown"
	}
	// Match flightdeck.store.hostname(): short name only, so the same box does not appear as two
	// hosts depending on whether mDNS appended a suffix.
	if i := strings.IndexByte(name, '.'); i > 0 {
		name = name[:i]
	}
	return name
}

func expandHome(p string) string {
	if !strings.HasPrefix(p, "~") {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return p
	}
	return filepath.Join(home, strings.TrimPrefix(p, "~"))
}

func nowMS() int64 { return time.Now().UnixMilli() }
