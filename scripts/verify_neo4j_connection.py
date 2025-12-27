"""
Neo4j 连接与读写验证脚本

用于验证 Neo4j 服务是否正常运行，以及适配器能否正确读写数据。
"""
import argparse
import sys
import os
import logging
import uuid
from pathlib import Path

# 添加项目根目录到 path
sys.path.append(str(Path(__file__).parent.parent))

from src.adapters.graph_store.neo4j_adapter import Neo4jAdapter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_env_file(env_file: str) -> None:
    if not env_file:
        return
    p = Path(env_file)
    if not p.exists():
        logger.warning(f"Neo4j env file not found: {env_file}")
        return
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read Neo4j env file: {e}")
        return

    for line in text.splitlines():
        s = (line or "").strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = (k or "").strip()
        v = (v or "").strip().strip('"').strip("'")
        if not k:
            continue
        if k not in os.environ:
            os.environ[k] = v
        if k == "NEO4J_USERNAME" and "NEO4J_USER" not in os.environ:
            os.environ["NEO4J_USER"] = v


def verify_neo4j(env_file: str = ""):
    load_env_file(env_file)
    logger.info("开始验证 Neo4j 连接与读写...")

    # 1. 初始化适配器
    try:
        neo4j = Neo4jAdapter()
        if not neo4j.is_available():
            logger.error("❌ 无法连接到 Neo4j。请检查连接信息与网络/服务状态。")
            return
        logger.info("✅ Neo4j 连接成功")
    except Exception as e:
        logger.error(f"❌ 初始化适配器失败: {e}")
        return

    # 生成唯一的测试 ID
    test_id = str(uuid.uuid4())
    test_node_label = "TestNode"
    
    try:
        # 2. 测试写入 (CREATE)
        logger.info(f"正在写入测试节点 (ID: {test_id})...")
        create_query = f"""
        CREATE (n:{test_node_label} {{id: $id, name: 'Test Node', created_at: timestamp()}})
        RETURN n
        """
        neo4j.query(create_query, {"id": test_id})
        logger.info("✅ 写入操作完成")

        # 3. 测试读取 (MATCH)
        logger.info("正在读取测试节点...")
        match_query = f"""
        MATCH (n:{test_node_label} {{id: $id}})
        RETURN n.name as name, n.created_at as created_at
        """
        results = neo4j.query(match_query, {"id": test_id})
        
        if results and len(results) == 1 and results[0]['name'] == 'Test Node':
            logger.info(f"✅ 读取成功: {results[0]}")
        else:
            logger.error(f"❌ 读取失败或数据不匹配: {results}")

        # 4. 测试批量执行 (Batch Execution)
        logger.info("正在测试批量执行...")
        batch_ops = [
            {
                "cypher": f"CREATE (n:{test_node_label}_Batch {{id: $id, idx: $idx}})",
                "params": {"id": test_id, "idx": i}
            }
            for i in range(3)
        ]
        neo4j.execute_batch(batch_ops)
        
        # 验证批量写入
        count_res = neo4j.query(f"MATCH (n:{test_node_label}_Batch {{id: $id}}) RETURN count(n) as cnt", {"id": test_id})
        if count_res and count_res[0]['cnt'] == 3:
             logger.info("✅ 批量执行成功")
        else:
             logger.error(f"❌ 批量执行验证失败: {count_res}")

    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
    finally:
        # 5. 清理数据 (DELETE)
        logger.info("正在清理测试数据...")
        try:
            cleanup_query = f"""
            MATCH (n) 
            WHERE n.id = $id AND (n:{test_node_label} OR n:{test_node_label}_Batch)
            DETACH DELETE n
            """
            neo4j.query(cleanup_query, {"id": test_id})
            logger.info("✅ 清理完成")
        except Exception as e:
            logger.error(f"❌ 清理数据失败: {e}")
        
        neo4j.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=os.getenv("NEO4J_ENV_FILE", ""))
    args = parser.parse_args()
    verify_neo4j(env_file=args.env_file)
