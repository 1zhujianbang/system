# src/agents/kg_tests.py
"""
知识图谱功能测试脚本

此脚本测试以下功能：
1. 知识图谱核心功能（构建、查询、更新）
2. 与现有系统的集成
3. 可视化和解释功能
"""

import os
import sys
import json
import time
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kg_tests")

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.tool_function import tools

# 测试类
class KGTester:
    """知识图谱测试器"""
    
    def __init__(self):
        self.results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "failures": []
        }
        self.start_time = None
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始知识图谱功能测试...")
        self.start_time = time.time()
        
        try:
            # 测试核心模块导入
            self.test_module_imports()
            
            # 测试知识图谱核心功能
            self.test_knowledge_graph_core()
            
            # 测试接口层
            self.test_kg_interface()
            
            # 测试可视化功能
            self.test_visualization()
            
            # 测试系统集成
            self.test_system_integration()
            
        except Exception as e:
            logger.error(f"测试过程中发生错误: {e}")
            self.results["tests_failed"] += 1
            self.results["failures"].append({"test": "run_all_tests", "error": str(e)})
        finally:
            self._print_summary()
    
    def _test(self, test_name, test_func):
        """执行单个测试"""
        logger.info(f"运行测试: {test_name}")
        try:
            test_func()
            logger.info(f"✅ 测试通过: {test_name}")
            self.results["tests_passed"] += 1
            return True
        except Exception as e:
            logger.error(f"❌ 测试失败: {test_name} - {e}")
            self.results["tests_failed"] += 1
            self.results["failures"].append({"test": test_name, "error": str(e)})
            return False
    
    def _print_summary(self):
        """打印测试结果摘要"""
        elapsed_time = time.time() - self.start_time
        total_tests = self.results["tests_passed"] + self.results["tests_failed"]
        
        logger.info("\n========== 测试结果摘要 ==========")
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {self.results['tests_passed']}")
        logger.info(f"失败测试: {self.results['tests_failed']}")
        logger.info(f"测试耗时: {elapsed_time:.2f} 秒")
        
        if self.results["failures"]:
            logger.error("\n失败详情:")
            for i, failure in enumerate(self.results["failures"], 1):
                logger.error(f"{i}. {failure['test']}: {failure['error']}")
        
        if self.results["tests_failed"] == 0:
            logger.info("🎉 所有测试通过！")
        else:
            logger.error("⚠️ 有测试失败，请检查错误详情")
    
    def test_module_imports(self):
        """测试所有模块是否能正确导入"""
        
        def _import_test():
            # 测试核心模块导入
            from src.agents.knowledge_graph import KnowledgeGraph
            from src.agents.kg_interface import get_knowledge_graph, refresh_graph
            from src.agents.kg_visualization import (
                KGVisualizer, KGExplainer, visualize_entities, 
                generate_report, get_graph_statistics
            )
            
            logger.info("所有模块导入成功")
        
        self._test("module_imports", _import_test)
    
    def test_knowledge_graph_core(self):
        """测试知识图谱核心功能"""
        
        def _core_test():
            from src.agents.knowledge_graph import KnowledgeGraph
            
            # 创建知识图谱实例
            kg = KnowledgeGraph()
            
            # 测试基本方法
            entities = kg.get_all_entities()
            events = kg.get_all_events()
            
            logger.info(f"知识图谱统计: 实体数={len(entities)}, 事件数={len(events)}")
            
            # 测试至少存在一些实体和事件
            # 注意：如果是首次运行，可能没有数据，这里不做强制要求
            if len(entities) > 0:
                sample_entity = next(iter(entities.keys()))
                entity_events = kg.get_entity_events(sample_entity)
                related_entities = kg.get_related_entities(sample_entity)
                
                logger.info(f"样本实体 '{sample_entity}' 的事件数: {len(entity_events)}")
                logger.info(f"样本实体 '{sample_entity}' 的相关实体数: {len(related_entities)}")
            
            # 测试图谱构建
            kg.build_graph()
            logger.info("知识图谱构建完成")
        
        self._test("knowledge_graph_core", _core_test)
    
    def test_kg_interface(self):
        """测试知识图谱接口层"""
        
        def _interface_test():
            from src.agents.kg_interface import (
                get_knowledge_graph, refresh_graph, 
                search_entities, search_events, get_entity_relations
            )
            
            # 测试单例模式
            kg1 = get_knowledge_graph()
            kg2 = get_knowledge_graph()
            assert kg1 is kg2, "知识图谱单例模式失败"
            
            # 测试刷新功能（异步，不等待完成）
            refresh_graph(force=True)
            logger.info("知识图谱刷新调用成功")
            
            # 测试搜索功能
            entities = kg1.get_all_entities()
            if entities:
                # 使用第一个实体的部分名称进行搜索
                sample_entity = next(iter(entities.keys()))
                search_term = sample_entity[:3]  # 使用前3个字符作为搜索词
                
                search_results = search_entities(search_term)
                logger.info(f"搜索 '{search_term}' 结果数: {len(search_results)}")
                
                # 测试关系查询
                relations = get_entity_relations(sample_entity)
                logger.info(f"实体关系查询结果数: {len(relations)}")
        
        self._test("kg_interface", _interface_test)
    
    def test_visualization(self):
        """测试可视化功能（不实际生成图像，只测试接口）"""
        
        def _visualization_test():
            from src.agents.kg_visualization import (
                KGVisualizer, KGExplainer, generate_report, 
                get_graph_statistics, detect_graph_anomalies
            )
            
            # 测试可视化器初始化
            visualizer = KGVisualizer()
            explainer = KGExplainer()
            
            # 测试统计信息获取
            stats = get_graph_statistics()
            logger.info(f"图谱统计: {stats}")
            
            # 测试报告生成（限制深度以避免性能问题）
            report = generate_report()
            logger.info(f"生成全局报告成功，包含 {len(report.get('key_entities', []))} 个关键实体")
            
            # 测试异常检测
            anomalies = detect_graph_anomalies()
            logger.info(f"异常检测发现 {len(anomalies)} 个异常")
        
        self._test("visualization", _visualization_test)
    
    def test_system_integration(self):
        """测试与现有系统的集成"""
        
        def _integration_test():
            # 检查数据文件是否存在
            required_files = [
                tools.ENTITIES_FILE,
                tools.ABSTRACT_MAP_FILE
            ]
            
            for file_path in required_files:
                if os.path.exists(file_path):
                    logger.info(f"数据文件存在: {file_path}")
                else:
                    logger.warning(f"数据文件不存在: {file_path}")
            
            # 验证agent1和agent2中的导入
            # 通过导入相关模块验证
            try:
                import src.agents.agent1
                import src.agents.agent2
                logger.info("agent1和agent2模块导入成功")
            except Exception as e:
                logger.warning(f"agent模块导入警告: {e}")
            
            # 测试知识图谱是否能从现有数据构建
            from src.agents.kg_interface import get_knowledge_graph
            kg = get_knowledge_graph()
            
            # 重新构建以确保能正确读取现有数据
            kg.build_graph()
            
            logger.info("系统集成测试完成")
        
        self._test("system_integration", _integration_test)

# 主函数
def run_tests():
    """运行测试入口"""
    tester = KGTester()
    tester.run_all_tests()
    return tester.results

# 如果直接运行此脚本
if __name__ == "__main__":
    run_tests()
