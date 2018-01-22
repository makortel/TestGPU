#ifndef TestGPU_AcceleratorService_interface_TestProxyProduct_h
#define TestGPU_AcceleratorService_interface_TestProxyProduct_h

#include "TestGPU/AcceleratorService/interface/AcceleratorTask.h"

class TestProxyProduct {
public:
  using TaskGetter = std::function<const AcceleratorTaskBase *(void)>;

  TestProxyProduct() = default;
  TestProxyProduct(int value): value_(value) {}
  TestProxyProduct(TaskGetter taskGetter): taskGetter_(taskGetter) {}
  ~TestProxyProduct() = default;

  int value() const { return value_; }
  const AcceleratorTaskBase *getTask() const { return taskGetter_(); }

private:
  int value_;

  TaskGetter taskGetter_;
};

#endif
