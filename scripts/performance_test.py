#!/usr/bin/env python3
"""
MarketLens 性能测试脚本

用于测试系统的性能并生成报告。
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import statistics

import aiohttp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8501"):
        self.base_url = base_url
        self.results = {
            "test_start_time": datetime.now().isoformat(),
            "test_cases": [],
            "summary": {}
        }
    
    async def test_api_endpoints(self) -> Dict[str, Any]:
        """测试API端点性能"""
        test_cases = [
            {
                "name": "Health Check",
                "endpoint": "/health",
                "method": "GET",
                "requests": 100
            },
            {
                "name": "Entity Search",
                "endpoint": "/api/entities/search",
                "method": "GET",
                "params": {"query": "China"},
                "requests": 50
            },
            {
                "name": "Event Search",
                "endpoint": "/api/events/search",
                "method": "GET",
                "params": {"query": "meeting"},
                "requests": 50
            }
        ]
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            for test_case in test_cases:
                case_result = await self._run_test_case(session, test_case)
                results.append(case_result)
                self.results["test_cases"].append(case_result)
        
        return results
    
    async def _run_test_case(self, session, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个测试用例"""
        name = test_case["name"]
        endpoint = test_case["endpoint"]
        method = test_case["method"]
        requests = test_case.get("requests", 10)
        params = test_case.get("params", {})
        
        print(f"Running test case: {name}")
        
        latencies = []
        errors = 0
        start_time = time.time()
        
        # 并发执行请求
        tasks = []
        for i in range(requests):
            task = self._make_request(session, f"{self.base_url}{endpoint}", method, params)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, Exception):
                errors += 1
                continue
            
            latency, success = response
            if success:
                latencies.append(latency)
            else:
                errors += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算统计数据
        stats = {
            "test_case": name,
            "total_requests": requests,
            "successful_requests": len(latencies),
            "error_requests": errors,
            "total_time": total_time,
            "requests_per_second": requests / total_time if total_time > 0 else 0,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "min_latency": min(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "latency_95th": self._percentile(latencies, 95) if latencies else 0,
            "latency_99th": self._percentile(latencies, 99) if latencies else 0
        }
        
        print(f"Completed test case: {name}")
        print(f"  Success rate: {stats['successful_requests']/requests*100:.2f}%")
        print(f"  Avg latency: {stats['avg_latency']:.2f}ms")
        print(f"  Requests/sec: {stats['requests_per_second']:.2f}")
        
        return stats
    
    async def _make_request(self, session, url: str, method: str, params: Dict[str, Any]) -> tuple:
        """发起HTTP请求"""
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                async with session.get(url, params=params) as response:
                    await response.read()
                    success = response.status == 200
            elif method.upper() == "POST":
                async with session.post(url, json=params) as response:
                    await response.read()
                    success = response.status == 200
            else:
                success = False
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            
            return latency, success
            
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            return latency, False
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def generate_load_test(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """生成负载测试"""
        print(f"Starting load test for {duration_minutes} minutes...")
        
        # 模拟用户行为
        user_actions = [
            {"endpoint": "/api/entities/search", "params": {"query": "China"}},
            {"endpoint": "/api/entities/search", "params": {"query": "USA"}},
            {"endpoint": "/api/events/search", "params": {"query": "meeting"}},
            {"endpoint": "/api/events/search", "params": {"query": "trade"}},
            {"endpoint": "/health", "params": {}}
        ]
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        request_count = 0
        error_count = 0
        
        async def simulate_user():
            nonlocal request_count, error_count
            async with aiohttp.ClientSession() as session:
                while time.time() < end_time:
                    action = random.choice(user_actions)
                    try:
                        async with session.get(f"{self.base_url}{action['endpoint']}", params=action['params']) as response:
                            await response.read()
                            if response.status == 200:
                                request_count += 1
                            else:
                                error_count += 1
                    except Exception:
                        error_count += 1
                    
                    # 随机等待0.1-1秒模拟用户思考时间
                    await asyncio.sleep(random.uniform(0.1, 1.0))
        
        # 模拟10个并发用户
        async def run_load_test():
            tasks = [simulate_user() for _ in range(10)]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # 运行负载测试
        asyncio.run(run_load_test())
        
        total_time = time.time() - start_time
        requests_per_second = request_count / total_time if total_time > 0 else 0
        
        load_test_result = {
            "test_type": "load_test",
            "duration_minutes": duration_minutes,
            "total_requests": request_count,
            "error_requests": error_count,
            "successful_requests": request_count - error_count,
            "total_time": total_time,
            "requests_per_second": requests_per_second,
            "success_rate": (request_count - error_count) / request_count * 100 if request_count > 0 else 0
        }
        
        self.results["test_cases"].append(load_test_result)
        
        print(f"Load test completed:")
        print(f"  Total requests: {request_count}")
        print(f"  Success rate: {load_test_result['success_rate']:.2f}%")
        print(f"  Requests/sec: {requests_per_second:.2f}")
        
        return load_test_result
    
    def generate_report(self) -> str:
        """生成性能测试报告"""
        self.results["test_end_time"] = datetime.now().isoformat()
        
        # 计算汇总统计
        total_requests = sum(case["total_requests"] for case in self.results["test_cases"] if "total_requests" in case)
        total_errors = sum(case["error_requests"] for case in self.results["test_cases"] if "error_requests" in case)
        total_success = total_requests - total_errors
        
        self.results["summary"] = {
            "total_test_cases": len([case for case in self.results["test_cases"] if "test_case" in case]),
            "total_requests": total_requests,
            "successful_requests": total_success,
            "error_requests": total_errors,
            "overall_success_rate": total_success / total_requests * 100 if total_requests > 0 else 0
        }
        
        # 生成报告文本
        report = []
        report.append("=" * 80)
        report.append("MARKETLENS PERFORMANCE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test Start Time: {self.results['test_start_time']}")
        report.append(f"Test End Time: {self.results['test_end_time']}")
        report.append("")
        
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Test Cases: {self.results['summary']['total_test_cases']}")
        report.append(f"Total Requests: {self.results['summary']['total_requests']}")
        report.append(f"Successful Requests: {self.results['summary']['successful_requests']}")
        report.append(f"Error Requests: {self.results['summary']['error_requests']}")
        report.append(f"Overall Success Rate: {self.results['summary']['overall_success_rate']:.2f}%")
        report.append("")
        
        report.append("DETAILED RESULTS")
        report.append("-" * 40)
        for case in self.results["test_cases"]:
            if "test_case" in case:
                report.append(f"Test Case: {case['test_case']}")
                report.append(f"  Requests: {case['total_requests']}")
                report.append(f"  Successful: {case['successful_requests']}")
                report.append(f"  Errors: {case['error_requests']}")
                report.append(f"  Requests/sec: {case['requests_per_second']:.2f}")
                report.append(f"  Avg Latency: {case['avg_latency']:.2f}ms")
                report.append(f"  95th Percentile: {case['latency_95th']:.2f}ms")
                report.append(f"  99th Percentile: {case['latency_99th']:.2f}ms")
            elif "test_type" in case and case["test_type"] == "load_test":
                report.append(f"Load Test ({case['duration_minutes']} minutes)")
                report.append(f"  Total Requests: {case['total_requests']}")
                report.append(f"  Successful Requests: {case['successful_requests']}")
                report.append(f"  Error Requests: {case['error_requests']}")
                report.append(f"  Success Rate: {case['success_rate']:.2f}%")
                report.append(f"  Requests/sec: {case['requests_per_second']:.2f}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "performance_report.json"):
        """保存测试结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


async def main():
    """主函数"""
    tester = PerformanceTester()
    
    print("Starting MarketLens Performance Tests...")
    
    # 测试API端点
    await tester.test_api_endpoints()
    
    # 运行负载测试
    tester.generate_load_test(duration_minutes=2)
    
    # 生成报告
    report = tester.generate_report()
    print(report)
    
    # 保存结果
    tester.save_results()
    
    # 保存报告到文件
    with open("performance_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\nPerformance testing completed!")


if __name__ == "__main__":
    asyncio.run(main())