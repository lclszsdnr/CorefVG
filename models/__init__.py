# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .coref_deter import build

'''
__init__.py 在Python2之前在package下必须设置，只有在文件夹下有该文件，Python才能知道
这是一个package，而不是一个普通的文件夹。
在import package的时候，__init__.py 会被自动执行，因此一般在该文件下进行一些初始化操作
在这里就是定义了一个函数
'''
def build_model(args):
    return build(args)
