package agilepool

import (
	"context"
	"log"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

const (
	defaultCleanPeriod          = 500 * time.Millisecond
	defaultTaskQueueSize        = 10000
	defaultMaxWorkerNumCapacity = math.MaxInt64
	defaultWorkMode             = BLOCK
	defaultIdleContainerType    = LinkedListType
)

type WorkMode int8

const (
	BLOCK WorkMode = iota
	NONBLOCK
)

// Logger defines the logging interface used by the pool.
// Both the standard library's *log.Logger and structured loggers
// (e.g. zap.SugaredLogger) satisfy this interface.
type Logger interface {
	Printf(format string, v ...interface{})
	Println(v ...interface{})
}

// defaultPool holds the most recently created Pool instance.
// It allows users to retrieve the pool object conveniently without
// keeping a reference themselves.
var defaultPool atomic.Pointer[Pool]

// GetDefaultPool returns the most recently created Pool, or nil if
// no pool has been created yet or the pool has been closed.
func GetDefaultPool() *Pool {
	p := defaultPool.Load()
	if p != nil && atomic.LoadInt32(&p.closed) == 1 {
		return nil
	}
	return p
}

type Pool struct {
	taskQueue         chan Task
	closePoolCn       chan struct{}
	capacity          int64 // The maximum number of workers in the pool.
	runningWorkersNum int64
	closed            int32 // 1 once Close has been called, otherwise 0
	muIdle            sync.Mutex
	workerPool        sync.Pool // Worker object pool
	idleWorks         IdleWorkerContainer
	config            *Config
	lock              *sync.Mutex
	wg                sync.WaitGroup
	logger            Logger
	// workerCreateCount counts the total allocations from sync.Pool.New
	// over the pool lifetime. For the number of currently active workers,
	// use GetRunningWorkersNum().
	workerCreateCount int64
}

func NewPool(c *Config) *Pool {
	if c == nil {
		c = NewConfig()
	}
	p := &Pool{
		closePoolCn: make(chan struct{}),
		config:      c,
		lock:        &sync.Mutex{},
		logger:      log.Default(),
		capacity:    c.workerNumCapacity,
		taskQueue:   make(chan Task, c.taskQueueSize),
	}

	switch c.idleContainerType {
	case MinHeapType:
		p.idleWorks = newMinHeap()
	case SliceType:
		p.idleWorks = newSlice()
	default:
		p.idleWorks = newLinkedList()
	}

	atomic.StoreInt64(&p.workerCreateCount, 0)

	p.workerPool.New = func() interface{} {
		atomic.AddInt64(&p.workerCreateCount, 1)
		w := &worker{
			pool: p,
		}
		return w
	}

	go p.expiredWorkerCleaner()
	defaultPool.Store(p)
	return p
}

// SetLogger replaces the default standard-library logger.
// Pass the same logger instance used elsewhere in your application
// (e.g. zap.SugaredLogger) so pool output appears in the same log stream.
func (p *Pool) SetLogger(l Logger) {
	p.logger = l
}

func (p *Pool) Submit(task Task) {
	// Reject new submissions once Close has been called. We check before
	// wg.Add so that a closed pool's Wait() can still return promptly for
	// in-flight tasks and is not blocked by post-close submissions.
	if atomic.LoadInt32(&p.closed) == 1 {
		return
	}
	p.wg.Add(1)

	// Fast path: claim a worker slot with a lock-free CAS instead of the
	// pool mutex. The CAS makes the capacity check + increment atomic, which
	// is all the mutex provided here; the stranded-task guards live in
	// ensureQueueConsumed and in the worker's locked exit protocol. Keeping
	// this path (and the common slow-path cases) off the mutex is what lets
	// throughput scale with large worker capacities instead of collapsing
	// under lock contention.
	for {
		running := atomic.LoadInt64(&p.runningWorkersNum)
		if running >= p.capacity {
			break
		}
		if !atomic.CompareAndSwapInt64(&p.runningWorkersNum, running, running+1) {
			continue
		}
		p.muIdle.Lock()
		w := p.idleWorks.Pop()
		p.muIdle.Unlock()
		if w == nil {
			w = p.workerPool.Get().(*worker)
		}
		go w.run(task)
		return
	}

	if p.config.workMode == NONBLOCK {
		p.wg.Done()
		return
	}

	p.taskQueue <- task
	p.ensureQueueConsumed()
}

// ensureQueueConsumed is the safety net against the stranded-task race: a
// task pushed to taskQueue right as the last workers decide to park could be
// left with no consumer. It must be called after every push to taskQueue.
//
// One lock-free fast-out covers the common case: if the queue is observed
// empty after the push, every queued task (ours included) has already been
// consumed by a live worker goroutine, so there is nothing left to strand.
// Our own send happens-before this len() read, so the read cannot predate
// the push.
//
// Note that a fast-out based on observing runningWorkersNum >= capacity is
// NOT sound: a parking worker re-checks the queue before it decrements the
// counter (both under lock), so a submitter can push after that re-check yet
// still read the not-yet-decremented counter and wrongly skip the net.
func (p *Pool) ensureQueueConsumed() {
	if len(p.taskQueue) == 0 {
		return
	}
	// Re-check under lock and spawn enough workers to drain the queue, up to
	// capacity. The lock serializes this decision against the workers' locked
	// park/exit protocol. See TestAgilePoolRaceStuckTaskInQueue for the race
	// this guards against.
	p.lock.Lock()
	target := int64(len(p.taskQueue))
	if target > p.capacity {
		target = p.capacity
	}
	// Claim each slot with a CAS rather than a blind add: Submit's lock-free
	// fast path increments runningWorkersNum concurrently without holding the
	// lock, so only a bounded CAS keeps the count from exceeding capacity.
	toSpawn := int64(0)
	for {
		running := atomic.LoadInt64(&p.runningWorkersNum)
		if running >= target {
			break
		}
		if atomic.CompareAndSwapInt64(&p.runningWorkersNum, running, running+1) {
			toSpawn++
		}
	}
	p.lock.Unlock()
	for i := int64(0); i < toSpawn; i++ {
		w := p.workerPool.Get().(*worker)
		go w.run(nil)
	}
}

// Submits a task before the specified timeout. If timeout is reached during execution, the task is canceled.
func (p *Pool) SubmitBefore(task Task, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	p.Submit(
		TaskFunc(func() error {
			defer cancel() // Ensures context is released after task completes to avoid resource leak
			select {
			case <-ctx.Done():
				return nil // Timeout reached, exit early
			default:
				task.Process() // Execute the task
			}
			return nil
		}),
	)
}

func (p *Pool) addToIdle(w *worker) {
	p.muIdle.Lock()
	defer p.muIdle.Unlock()
	p.idleWorks.Add(w)
}

func (p *Pool) addRunningWorkersNum(num int64) {
	atomic.AddInt64(&p.runningWorkersNum, num)
}

func (p *Pool) expiredWorkerCleaner() {
	ticker := time.NewTicker(p.config.cleanPeriod)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			p.muIdle.Lock()
			p.idleWorks.RemoveExpired(time.Now(), 1*time.Second)
			p.muIdle.Unlock()
			runtime.Gosched()
		case <-p.closePoolCn:
			return
		}
	}
}

// Close marks the pool as closed and stops its background cleaner goroutine.
// After Close:
//   - new Submit calls become no-ops (the task is dropped, no goroutine is started)
//   - in-flight tasks already submitted continue to run to completion
//   - Wait() returns once all in-flight tasks are done, enabling graceful shutdown
//
// Close is idempotent and safe to call from any goroutine, including from
// within a running task.
func (p *Pool) Close() {
	if !atomic.CompareAndSwapInt32(&p.closed, 0, 1) {
		return
	}
	defaultPool.CompareAndSwap(p, nil)
	close(p.closePoolCn)
}

func (p *Pool) Wait() {
	p.wg.Wait()
}

func (p *Pool) done() {
	p.wg.Done()
}

func (p *Pool) GetRunningWorkersNum() int64 {
	return atomic.LoadInt64(&p.runningWorkersNum)
}

// GetWorkerCreateCount returns the total number of worker structs that have
// been allocated from sync.Pool.New over the pool's lifetime.
func (p *Pool) GetWorkerCreateCount() int64 {
	return atomic.LoadInt64(&p.workerCreateCount)
}
