#ifndef __CPM_HPP__
#define __CPM_HPP__

// Comsumer Producer Model

#include <algorithm>
#include <condition_variable>
#include <future>
#include <memory>
#include <queue>
#include <thread>

namespace cpm {

template <typename Result, typename Input, typename Model>
class Instance {
 protected:
  struct Item {
    Input input;
    std::shared_ptr<std::promise<Result>> pro;
  };

  std::condition_variable cond_;
  std::queue<Item> input_queue_;
  std::mutex queue_lock_;
  std::shared_ptr<std::thread> worker_;
  volatile bool run_ = false;
  volatile int max_items_processed_ = 0;
  void *stream_ = nullptr;

 public:
  virtual ~Instance() { stop(); }

  void stop() {
    run_ = false;
    cond_.notify_one();
    {
      std::unique_lock<std::mutex> l(queue_lock_);
      while (!input_queue_.empty()) {
        auto &item = input_queue_.front();
        if (item.pro) item.pro->set_value(Result());
        input_queue_.pop();
      }
    };

    if (worker_) {
      worker_->join();
      worker_.reset();
    }
  }

  virtual std::shared_future<Result> commit(const Input &input) {
    Item item;
    item.input = input;
    item.pro.reset(new std::promise<Result>());
    {
      std::unique_lock<std::mutex> __lock_(queue_lock_);
      input_queue_.push(item);
    }
    cond_.notify_one();
    return item.pro->get_future();
  }

  virtual std::vector<std::shared_future<Result>> commits(const std::vector<Input> &inputs) {
    std::vector<std::shared_future<Result>> output;
    {
      std::unique_lock<std::mutex> __lock_(queue_lock_);
      for (int i = 0; i < (int)inputs.size(); ++i) {
        Item item;
        item.input = inputs[i];
        item.pro.reset(new std::promise<Result>());
        output.emplace_back(item.pro->get_future());
        input_queue_.push(item);
      }
    }
    cond_.notify_one();
    return output;
  }

  template <typename LoadMethod>
  bool start(const LoadMethod &loadmethod, int max_items_processed = 1, void *stream = nullptr) {
    stop();

    this->stream_ = stream;
    this->max_items_processed_ = max_items_processed;
    std::promise<bool> status;
    worker_ = std::make_shared<std::thread>(&Instance::worker<LoadMethod>, this,
                                            std::ref(loadmethod), std::ref(status));
    return status.get_future().get();
  }

 private:
  template <typename LoadMethod>
  void worker(const LoadMethod &loadmethod, std::promise<bool> &status) {
    std::shared_ptr<Model> model = loadmethod();
    if (model == nullptr) {
      status.set_value(false);
      return;
    }

    run_ = true;
    status.set_value(true);

    std::vector<Item> fetch_items;
    std::vector<Input> inputs;
    while (get_items_and_wait(fetch_items, max_items_processed_)) {
      inputs.resize(fetch_items.size());
      std::transform(fetch_items.begin(), fetch_items.end(), inputs.begin(),
                     [](Item &item) { return item.input; });

      auto ret = model->forwards(inputs, stream_);
      for (int i = 0; i < (int)fetch_items.size(); ++i) {
        if (i < (int)ret.size()) {
          fetch_items[i].pro->set_value(ret[i]);
        } else {
          fetch_items[i].pro->set_value(Result());
        }
      }
      inputs.clear();
      fetch_items.clear();
    }
    model.reset();
    run_ = false;
  }

  virtual bool get_items_and_wait(std::vector<Item> &fetch_items, int max_size) {
    std::unique_lock<std::mutex> l(queue_lock_);
    cond_.wait(l, [&]() { return !run_ || !input_queue_.empty(); });

    if (!run_) return false;

    fetch_items.clear();
    for (int i = 0; i < max_size && !input_queue_.empty(); ++i) {
      fetch_items.emplace_back(std::move(input_queue_.front()));
      input_queue_.pop();
    }
    return true;
  }

  virtual bool get_item_and_wait(Item &fetch_item) {
    std::unique_lock<std::mutex> l(queue_lock_);
    cond_.wait(l, [&]() { return !run_ || !input_queue_.empty(); });

    if (!run_) return false;

    fetch_item = std::move(input_queue_.front());
    input_queue_.pop();
    return true;
  }
};
};  // namespace cpm

#endif  // __CPM_HPP__