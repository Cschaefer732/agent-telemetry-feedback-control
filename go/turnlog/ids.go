package turnlog

import (
	"crypto/rand"
	"sync"
	"time"
)

// ULIDs, matching flightdeck/ids.py. Turn ids are sorted by time constantly in both the sqlite
// index and in raw log greps; lexicographic order matching chronological order is worth the
// 26 characters over a UUID.

const crockford = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

var (
	idMu     sync.Mutex
	idLastMS int64
	idLastRd uint64
)

func encodeBase32(value uint64, length int) string {
	out := make([]byte, length)
	for i := length - 1; i >= 0; i-- {
		out[i] = crockford[value&0x1F]
		value >>= 5
	}
	return string(out)
}

// NewID returns a 26-character ULID, monotonic within a process even inside the same millisecond.
func NewID() string {
	ms := time.Now().UnixMilli()
	idMu.Lock()
	var rd uint64
	if ms == idLastMS {
		idLastRd++
		rd = idLastRd
	} else {
		var buf [8]byte
		if _, err := rand.Read(buf[:]); err != nil {
			// Falling back to the nanosecond clock keeps ids unique-enough rather than returning
			// an error nobody at the call site can act on.
			rd = uint64(time.Now().UnixNano())
		} else {
			for _, b := range buf {
				rd = rd<<8 | uint64(b)
			}
		}
		idLastMS, idLastRd = ms, rd
	}
	idMu.Unlock()
	return encodeBase32(uint64(ms), 10) + encodeBase32(rd&((1<<50)-1), 10) + encodeBase32(rd>>50, 6)
}
