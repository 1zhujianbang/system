"""
GDELT数据源适配器模块单元测试
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime

from src.adapters.news.gdelt_adapter import GDELTAdapter
from src.ports.extraction import FetchConfig


class TestGDELTAdapter(unittest.TestCase):
    
    def setUp(self):
        """测试初始化"""
        self.adapter = GDELTAdapter(name="TestGDELT")
        
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.adapter.source_name, "TestGDELT")
        self.assertTrue(self.adapter.is_available())
        
    def test_source_type(self):
        """测试数据源类型"""
        from src.ports.extraction import NewsSourceType
        self.assertEqual(self.adapter.source_type, NewsSourceType.CUSTOM)
        
    @patch('aiohttp.ClientSession')
    def test_fetch_success(self, mock_session_class):
        """测试成功获取数据"""
        # 模拟HTTP响应
        mock_response = Mock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="1234567890\t20230101\thttp://data.gdeltproject.org/gdeltv2/20230101.export.CSV.zip")
        
        # 创建模拟的上下文管理器
        mock_context_manager = Mock()
        mock_context_manager.__aenter__ = Mock(return_value=mock_response)
        mock_context_manager.__aexit__ = Mock(return_value=None)
        
        mock_session = Mock()
        mock_session.get = Mock(return_value=mock_context_manager)
        mock_session_class.return_value = mock_session
        
        # 准备测试配置
        config = FetchConfig(max_items=1)
        
        # 运行异步测试
        async def run_test():
            result = await self.adapter.fetch(config)
            return result
            
        result = asyncio.run(run_test())
        
        # 验证结果
        self.assertTrue(result.success)
        self.assertEqual(result.total_fetched, 1)
        self.assertEqual(len(result.items), 1)
        
    @patch('aiohttp.ClientSession')
    def test_fetch_failure(self, mock_session_class):
        """测试获取数据失败"""
        # 模拟HTTP错误响应
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        
        # 创建模拟的上下文管理器
        mock_context_manager = Mock()
        mock_context_manager.__aenter__ = Mock(return_value=mock_response)
        mock_context_manager.__aexit__ = Mock(return_value=None)
        
        mock_session = Mock()
        mock_session.get = Mock(return_value=mock_context_manager)
        mock_session_class.return_value = mock_session
        
        # 准备测试配置
        config = FetchConfig(max_items=1)
        
        # 运行异步测试
        async def run_test():
            result = await self.adapter.fetch(config)
            return result
            
        result = asyncio.run(run_test())
        
        # 验证结果
        self.assertFalse(result.success)
        self.assertIn("Failed to fetch master file list", result.error)
        
    def test_parse_master_file_list(self):
        """测试解析主文件列表"""
        # 准备测试数据
        content = "1234567890\t20230101\thttp://data.gdeltproject.org/gdeltv2/20230101.export.CSV.zip\n" \
                  "1234567891\t20230102\thttp://data.gdeltproject.org/gdeltv2/20230102.export.CSV.zip"
        config = FetchConfig(max_items=1)
        
        # 调用方法
        result = self.adapter._parse_master_file_list(content, config)
        
        # 验证结果
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        # 注意：由于是反向遍历，应该返回最新的文件
        self.assertEqual(result[0], "http://data.gdeltproject.org/gdeltv2/20230102.export.CSV.zip")


if __name__ == '__main__':
    unittest.main()