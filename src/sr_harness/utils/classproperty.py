# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
class classproperty(object):
    """
    自定义的类属性装饰器。
    允许通过 ClassName.property_name 直接获取动态计算的类属性，
    同时也支持 instance.property_name。
    """
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_instance, owner_class):
        # 无论通过实例调用还是类调用，统一将当前的 class 传给底层函数
        return self.fget(owner_class)
