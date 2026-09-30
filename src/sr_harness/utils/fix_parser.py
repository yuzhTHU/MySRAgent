# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import argparse

def add_minus_flags(parser: argparse.ArgumentParser):
    """
    自动为 argparse.ArgumentParser 中含有下划线的参数添加别名。例如：
    --fix_existing -> 添加别名 --fix-existing
    --augment_OD_num -> 添加别名 --augment-OD-num
    """
    for action in parser._actions:
        # 仅处理可选参数（以 '-' 开头的参数）
        if not action.option_strings:
            continue
        aliases_to_add = []
        for opt in action.option_strings:
            if opt.startswith('--') and '_' in opt:
                new_alias = opt.replace('_', '-')
                if new_alias not in action.option_strings:
                    aliases_to_add.append(new_alias)
        for alias in aliases_to_add:
            action.option_strings.append(alias)
            parser._option_string_actions[alias] = action
    return parser


def add_negation_flags(parser: argparse.ArgumentParser):
    """
    自动为 parser 中的 store_true 参数添加对应的 --no-xxx 选项。
    """
    args_to_add = []
    
    # 建立现有 flag 索引，防止重复添加
    existing_flags = set()
    for action in parser._actions:
        existing_flags.update(action.option_strings)
    for action in parser._actions:
        if isinstance(action, argparse._StoreTrueAction):
            # 为当前 action 收集所有可能的否定别名
            negation_aliases = []
            for option in action.option_strings:
                if not option.startswith('--'): 
                    continue
                # 生成否定形式：--use_new_model -> --no-use_new_model
                raw_name = option.removeprefix('--')
                neg_opt = f'--no-{raw_name}'
                # 只有当这个 flag 还不存在时才添加
                if neg_opt not in existing_flags:
                    negation_aliases.append(neg_opt)
            # 如果生成了有效的否定别名，将它们打包准备添加
            if negation_aliases:
                # 选取第一个选项名作为帮助文档显示的名称
                primary_name = action.option_strings[0].removeprefix('--')
                args_to_add.append({
                    'options': negation_aliases,  # 这里是一个列表
                    'dest': action.dest,
                    'action': 'store_false',
                    'default': action.default,
                    'help': f"Disable {primary_name}"
                })
    for arg in args_to_add:
        parser.add_argument(
            *arg['options'], 
            dest=arg['dest'],
            action=arg['action'],
            default=arg['default'],
            help=arg['help']
        )
    return parser
