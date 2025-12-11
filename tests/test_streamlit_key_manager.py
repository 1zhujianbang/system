#!/usr/bin/env python3
"""
Streamlit Key 管理器测试
"""

import pytest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.web.streamlit_key_manager import (
    StreamlitKeyManager,
    get_unique_key,
    push_context,
    pop_context,
    KeyContext
)


class TestStreamlitKeyManager:
    """Key管理器测试"""

    def setup_method(self):
        """每个测试前的设置"""
        self.manager = StreamlitKeyManager()

    def test_generate_unique_key_basic(self):
        """测试基本key生成"""
        key1 = self.manager.generate_key('gnews', 'category', context='config_tab')
        assert key1 == 'gnews_category_config_tab'

        # 再次生成应该不同
        key2 = self.manager.generate_key('gnews', 'category', context='config_tab')
        assert key2 == 'gnews_category_config_tab_1'

    def test_generate_key_with_index(self):
        """测试带索引的key生成"""
        key = self.manager.generate_key('data', 'row', index=5, context='table')
        assert key == 'data_row_table_5'

    def test_context_management(self):
        """测试上下文管理"""
        # 初始上下文
        assert self.manager.get_current_context() == 'global'

        # 推入上下文
        self.manager.push_context('config')
        assert self.manager.get_current_context() == 'config'

        self.manager.push_context('tab')
        assert self.manager.get_current_context() == 'config_tab'

        # 生成带上下文的key
        key = self.manager.generate_key('gnews', 'category')
        assert 'config_tab' in key

        # 弹出上下文
        popped = self.manager.pop_context()
        assert popped == 'tab'
        assert self.manager.get_current_context() == 'config'

    def test_key_registration(self):
        """测试key注册"""
        # 注册新key
        assert self.manager.register_key('test_key_1') == True

        # 重复注册应该失败
        assert self.manager.register_key('test_key_1') == False

        # 检查是否已使用
        assert self.manager.is_key_used('test_key_1') == True
        assert self.manager.is_key_used('unused_key') == False

    def test_context_manager(self):
        """测试上下文管理器"""
        assert self.manager.get_current_context() == 'global'

        with KeyContext('test_context'):
            assert self.manager.get_current_context() == 'test_context'

            key = self.manager.generate_key('test', 'element')
            assert 'test_context' in key

        # 退出后恢复
        assert self.manager.get_current_context() == 'global'

    def test_stats(self):
        """测试统计功能"""
        stats = self.manager.get_stats()
        assert 'used_keys_count' in stats
        assert 'current_context' in stats
        assert 'context_stack_depth' in stats

        # 添加一些key
        self.manager.generate_key('test', 'key1')
        self.manager.generate_key('test', 'key2')

        stats = self.manager.get_stats()
        assert stats['used_keys_count'] >= 2


class TestConvenienceFunctions:
    """便捷函数测试"""

    def setup_method(self):
        """重置全局管理器"""
        from src.web.streamlit_key_manager import key_manager
        key_manager.used_keys.clear()
        key_manager.context_stack.clear()

    def test_get_unique_key(self):
        """测试便捷函数"""
        key1 = get_unique_key('gnews', 'category', context='test')
        key2 = get_unique_key('gnews', 'category', context='test')

        assert key1 == 'gnews_category_test'
        assert key2 == 'gnews_category_test_1'

    def test_context_functions(self):
        """测试上下文函数"""
        from src.web.streamlit_key_manager import key_manager

        assert key_manager.get_current_context() == 'global'

        push_context('test')
        assert key_manager.get_current_context() == 'test'

        popped = pop_context()
        assert popped == 'test'
        assert key_manager.get_current_context() == 'global'


if __name__ == "__main__":
    pytest.main([__file__])
