#ifndef EDGETPU_CPP_PIPELINE_INTERNAL_THREAD_SAFE_QUEUE_H_
#define EDGETPU_CPP_PIPELINE_INTERNAL_THREAD_SAFE_QUEUE_H_

#include <algorithm>
#include <deque>
#include <functional>
#include <queue>
#include <string>

#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace coral {
namespace internal {

// WaitQueue supports both locking and waiting for new elements to appear.
//
// In particular, you can set the max size of the queue, see comments on
// set_max_queue_size for details.
//
// Example usage:
//
// - Create a WaitQueue q of elements of type T and start producer and/or
//   consumer threads. The main thread may also serve as a producer.
//
// - Producer calls q.push and/or q.push_front whenever it has data available.
//   q.push gives you FIFO queue semantics, while q.push_front gives you stack
//   semantics.
//
// - Each consumer sits in a loop such as:
//
//   T elt;
//   while (q.Wait(&elt)) {
//     ... consume elt ...
//   }
//   // Optionally notify that work is done
//
// - After the last element has been produced and pushed, call q.StopWaiters.
//   This will make each consumer exit its loop after all elements currently in
//   the queue have been consumed.  Do NOT push more elements after calling
//   q.StopWaiters because there may be no one left to consume them.
//
// All operations are atomic.
template <class T>
class WaitQueue {
 public:
  typedef std::deque<T> container_type;

  typedef typename container_type::value_type value_type;
  typedef typename container_type::size_type size_type;

  WaitQueue() : max_queue_size_(kInfiniteQueueSize), stop_requested_(false) {}

  bool empty() const {
    absl::ReaderMutexLock l(&busy_);
    return q_.empty();
  }
  size_type size() const {
    absl::ReaderMutexLock l(&busy_);
    return q_.size();
  }

  // Call to set a limited queue size. If this size is reached, calls
  // to push and push_front will block until elements have been popped
  // from the queue. Call with kInfiniteQueueSize (the default) to
  // make push and push_front non-blocking. This can be called while
  // the queue is active. If it is called with a size less than the
  // current queue size, future inserts will block until the queue
  // sizes goes below the new maximum.

  // size_type is always unsigned, so this is effectively infinite
  static const size_type kInfiniteQueueSize = static_cast<size_type>(-1);
  void set_max_queue_size(size_type max_queue_size) {
    absl::MutexLock l(&busy_);
    max_queue_size_ = max_queue_size;
    unfull_.SignalAll();
  }

  size_type max_queue_size() const {
    absl::MutexLock l(&busy_);
    return max_queue_size_;
  }

  // Push x onto the back of the queue.  The last item pushed this way
  // will be the last one to be popped.  If a waiter was waiting for
  // an element to appear, wake it up.  Will block if max_queue_size_
  // has been reached.
  void push(const value_type &x) {
    absl::MutexLock l(&busy_);
    while (q_.size() >= max_queue_size_) unfull_.Wait(&busy_);
    if (q_.empty()) ready_.Signal();
    q_.push_back(x);
  }

  // Same as push() but returns false if max_queue_size_ is reached instead of
  // blocking.  Returns true if x is pushed onto the queue.
  bool push_nowait(const value_type &x) {
    absl::MutexLock l(&busy_);
    if (q_.size() >= max_queue_size_) return false;
    if (q_.empty()) ready_.Signal();
    q_.push_back(x);
    return true;
  }

  // Push x onto the front of the queue.  The last item pushed this way
  // will be the first one to be popped.  If a waiter was waiting for an
  // element to appear, wake it up.  Will block if max_queue_size_
  // has been reached.
  void push_front(const value_type &x) {
    absl::MutexLock l(&busy_);
    while (q_.size() >= max_queue_size_) unfull_.Wait(&busy_);
    if (q_.empty()) ready_.Signal();
    q_.push_front(x);
  }

  // Same as push_front() but returns false if max_queue_size_ is reached
  // instead of blocking.  Returns true if x is pushed onto the queue.
  bool push_front_nowait(const value_type &x) {
    absl::MutexLock l(&busy_);
    if (q_.size() >= max_queue_size_) return false;
    if (q_.empty()) ready_.Signal();
    q_.push_front(x);
    return true;
  }

  // Atomically pop the front element into *p.  If it was present, return
  // true; if the queue was empty, leave *p unchanged and return false.
  bool Pop(value_type *p);

  // Atomically set *p to the front element.  If there was a front element,
  // return true; if the queue was empty, leave *p unchanged and return false.
  bool Front(value_type *p) const;

  // Atomically swap the container in the queue with the user provided
  // container, which should be empty. A common use case for this is to
  // send out everything in the queue in one operation.
  void SwapEmptyContainer(container_type *container) {
    DCHECK(container->empty());
    absl::MutexLock l(&busy_);
    q_.swap(*container);
    // In Pop(), each time we send a signal, a pending push() may be unblocked.
    // When push() is finished, the queue is guaranteed to be full again; the
    // next Pop() will unblock any remaining pending push(). So, calling
    // Signal() is sufficient there. This is not the case for
    // SwapEmptyContainer, because multiple elements can be removed from the
    // queue; we need to signal all the pending push().
    unfull_.SignalAll();
  }

  // Wait for a front element to appear.  If an element is ready, pop it into *p
  // and return true.  If the queue is currently empty, wait for an element to
  // appear, and then pop it into *p and return true.  Return false if the queue
  // is empty and StopWaiters() was called during or prior to waiting, in which
  // case *p is unchanged.
  bool Wait(value_type *p);

  // Wait for a front element to appear.  If an element is ready, pop it into
  // *p, clear *timed_out, and return true.  If the queue is currently empty,
  // wait for an element to appear and then pop it into *p, clear *timed_out,
  // and return true.  If the given timeout elapses without an element
  // appearing, set *timed_out and return true.  Return false if the queue is
  // empty and StopWaiters() was called during or prior to waiting, in which
  // case *p is unchanged.
  bool WaitWithTimeout(value_type *p, ::absl::Duration timeout,
                       bool *timed_out);

  // Terminate existing and future *blocking* Wait() requests.  Calls to Wait()
  // when the queue is non-empty are non-blocking, and so those are *not*
  // terminated.
  void StopWaiters() {
    absl::MutexLock l(&busy_);
    stop_requested_ = true;
    ready_.SignalAll();
  }

  // Atomically copy the current contents of the queue into a separate
  // container.
  //
  // IMPORTANT: For this method to be safe, item_type must be a type with value
  // semantics. Queues containing raw pointers must not be copied, because the
  // item might be popped and deleted before the copy is inspected. It is safe
  // to use shared_ptr<> items, but be aware that the items might be available
  // to two threads at once.
  void CopyTo(container_type *container) const {
    absl::MutexLock l(&busy_);
    *container = q_;
  }

 protected:
  mutable absl::Mutex busy_;
  container_type q_;
  // This condition variable is used to signal transitions from an empty to
  // a non-empty queue.
  absl::CondVar ready_;
  size_type max_queue_size_;

  // This condition is signaled whenever an element is removed from
  // the queue. Inserts block on this when (q_.size() >= max_queue_size_).
  absl::CondVar unfull_;
  bool stop_requested_;  // True after StopWaiters() has been called.
};

// ----------------------------------------------------------------------------
// Implementations of non-inline functions

template <class T>
bool WaitQueue<T>::Pop(value_type *p) {
  absl::MutexLock l(&busy_);
  if (q_.empty()) return false;
  *p = q_.front();
  q_.pop_front();
  unfull_.Signal();
  return true;
}

template <class T>
bool WaitQueue<T>::Front(value_type *p) const {
  absl::ReaderMutexLock l(&busy_);
  if (q_.empty()) return false;
  *p = q_.front();
  return true;
}

template <class T>
bool WaitQueue<T>::Wait(value_type *p) {
  absl::MutexLock l(&busy_);
  bool woken = false;
  while (q_.empty()) {
    if (stop_requested_) return false;
    ready_.Wait(&busy_);
    woken = true;
  }
  *p = q_.front();
  q_.pop_front();
  // Handle the case where more than one thread is waiting for a new entry
  // and two or more entries appear in rapid succession.
  if (woken && !q_.empty()) ready_.Signal();
  unfull_.Signal();
  return true;
}

template <class T>
bool WaitQueue<T>::WaitWithTimeout(value_type *p, ::absl::Duration timeout,
                                   bool *timed_out) {
  *timed_out = false;
  ::absl::Time deadline = ::absl::Now() + timeout;

  absl::MutexLock l(&busy_);
  bool woken = false;
  while (q_.empty()) {
    if (stop_requested_) return false;
    if (deadline <= ::absl::Now()) {
      *timed_out = true;
      return true;
    }
    ready_.WaitWithDeadline(&busy_, deadline);
    woken = true;
  }
  *p = q_.front();
  q_.pop_front();
  // Handle the case where more than one thread is waiting for a new entry
  // and two or more entries appear in rapid succession.
  if (woken && !q_.empty()) ready_.Signal();
  unfull_.Signal();
  return true;
}

}  // namespace internal
}  // namespace coral

#endif  // EDGETPU_CPP_PIPELINE_INTERNAL_THREAD_SAFE_QUEUE_H_
