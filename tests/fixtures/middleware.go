package middleware

import (
	"net/http"
	"strings"
	"sync"
	"time"
)

// RateLimitMiddleware enforces per-IP request throttling.
func RateLimitMiddleware(next http.Handler, requestsPerMinute int) http.Handler {
	var mu sync.Mutex
	clients := make(map[string][]time.Time)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ip := strings.Split(r.RemoteAddr, ":")[0]

		mu.Lock()
		now := time.Now()
		cutoff := now.Add(-time.Minute)

		// Prune old entries
		valid := make([]time.Time, 0)
		for _, t := range clients[ip] {
			if t.After(cutoff) {
				valid = append(valid, t)
			}
		}
		valid = append(valid, now)
		clients[ip] = valid
		count := len(valid)
		mu.Unlock()

		if count > requestsPerMinute {
			http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// AuthMiddleware validates Bearer tokens in the Authorization header.
func AuthMiddleware(next http.Handler, validateToken func(string) bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		token := auth[7:]
		if !validateToken(token) {
			http.Error(w, "Invalid token", http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
}
