"""
Neo4j 适配器测试
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# 添加项目根目录到 path
sys.path.append(str(Path(__file__).parent.parent))

from src.adapters.graph_store.neo4j_adapter import Neo4jAdapter

@pytest.fixture
def mock_driver():
    with patch("src.adapters.graph_store.neo4j_adapter.GraphDatabase.driver") as mock:
        driver_instance = MagicMock()
        mock.return_value = driver_instance
        yield driver_instance

def test_connect(mock_driver):
    adapter = Neo4jAdapter(uri="bolt://localhost:7687", user="neo4j", password="password")
    assert adapter._driver == mock_driver
    mock_driver.verify_connectivity.assert_called_once()

def test_is_available_true(mock_driver):
    adapter = Neo4jAdapter()
    assert adapter.is_available() is True

def test_is_available_false(mock_driver):
    mock_driver.verify_connectivity.side_effect = Exception("Connection failed")
    adapter = Neo4jAdapter()
    assert adapter.is_available() is False

def test_query(mock_driver):
    adapter = Neo4jAdapter()
    mock_session = MagicMock()
    mock_driver.session.return_value = mock_session
    mock_result = MagicMock()
    mock_record = MagicMock()
    mock_record.data.return_value = {"key": "value"}
    mock_result.__iter__.return_value = [mock_record]
    mock_session.__enter__.return_value.run.return_value = mock_result

    result = adapter.query("MATCH (n) RETURN n")
    assert result == [{"key": "value"}]
    mock_session.__enter__.return_value.run.assert_called_with("MATCH (n) RETURN n", {})

def test_execute_batch(mock_driver):
    adapter = Neo4jAdapter()
    mock_session = MagicMock()
    mock_driver.session.return_value = mock_session
    mock_session.__enter__.return_value = mock_session
    
    mock_tx = MagicMock()
    mock_tx_ctx = MagicMock()
    mock_session.begin_transaction.return_value = mock_tx_ctx
    mock_tx_ctx.__enter__.return_value = mock_tx

    operations = [
        {"cypher": "CREATE (n:Test {id: $id})", "params": {"id": 1}},
        {"cypher": "CREATE (n:Test {id: $id})", "params": {"id": 2}},
    ]
    adapter.execute_batch(operations)
    
    assert mock_tx.run.call_count == 2
    mock_tx_ctx.__exit__.assert_called()
