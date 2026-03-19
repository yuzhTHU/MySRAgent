"""数据统计分析工具的单元测试。"""

import numpy as np
import pytest

from sr_agent.tools.statistics import StatisticsTool


class TestStatisticsTool:
    """测试 StatisticsTool 的正确性。"""

    def setup_method(self):
        """每个测试方法前执行。"""
        # 测试数据
        self.x = {"feature1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        self.y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        self.tool = StatisticsTool(x=self.x, y=self.y)

    def test_basic_statistics(self):
        """测试基本统计量计算。"""
        result = self.tool.execute()

        # 检查返回结构
        assert "target" in result
        assert "features" in result
        assert len(result["features"]) == 1

        # 检查目标变量统计量
        target = result["target"]
        assert target["name"] == "y"
        assert target["n_samples"] == 5
        assert target["min"] == 2.0
        assert target["max"] == 10.0
        assert target["mean"] == 6.0
        assert target["median"] == 6.0

        # 检查特征统计量
        feature = result["features"][0]
        assert feature["name"] == "feature1"
        assert feature["n_samples"] == 5
        assert feature["min"] == 1.0
        assert feature["max"] == 5.0
        assert feature["mean"] == 3.0

    def test_multiple_features(self):
        """测试多特征输入。"""
        x = {
            "f1": np.array([1.0, 2.0, 3.0]),
            "f2": np.array([10.0, 20.0, 30.0]),
            "f3": np.array([100.0, 200.0, 300.0]),
        }
        y = np.array([5.0, 10.0, 15.0])
        tool = StatisticsTool(x=x, y=y)

        result = tool.execute()

        assert len(result["features"]) == 3
        feature_names = {f["name"] for f in result["features"]}
        assert feature_names == {"f1", "f2", "f3"}

    def test_variance_and_std(self):
        """测试方差和标准差计算。"""
        x = {"x": np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])}
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        tool = StatisticsTool(x=x, y=y)

        result = tool.execute()

        # 验证方差 (variance = sum((x - mean)^2) / n)
        expected_var_y = np.var([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert abs(result["target"]["variance"] - expected_var_y) < 1e-10
        assert abs(result["target"]["std"] - np.sqrt(expected_var_y)) < 1e-10

    def test_quartiles(self):
        """测试四分位数计算。"""
        x = {"x": np.array(range(1, 101))}  # 1 到 100
        y = np.array(range(1, 101))
        tool = StatisticsTool(x=x, y=y)

        result = tool.execute()

        # 对于 1-100，Q1=25.75, Q3=75.25 (线性插值)
        assert abs(result["target"]["q1"] - 25.75) < 0.01
        assert abs(result["target"]["q3"] - 75.25) < 0.01

    def test_callable(self):
        """测试工具可以像函数一样调用。"""
        result = self.tool()
        assert result["target"]["mean"] == 6.0

    def test_empty_input_raises(self):
        """测试空输入是否抛出异常。"""
        x = {"x": np.array([])}
        y = np.array([])
        tool = StatisticsTool(x=x, y=y)

        with pytest.raises(ValueError):
            # 空数组的统计量计算可能会出错
            tool.execute()

    def test_numpy_compatibility(self):
        """测试与 numpy 数组的兼容性。"""
        # 测试二维数组（应该被 flatten）
        x = {"x": np.array([[1.0], [2.0], [3.0]])}
        y = np.array([[1.0], [2.0], [3.0]])
        tool = StatisticsTool(x=x, y=y)

        result = tool.execute()
        assert result["target"]["n_samples"] == 3

    def test_metadata_exists(self):
        """测试元数据存在。"""
        assert self.tool.metadata is not None
        assert self.tool.metadata.name == "statistics_analysis"
        assert self.tool.metadata.category == "statistics"
        assert "统计量" in self.tool.metadata.description

    def test_x_vars_subset(self):
        """测试 x_vars 参数可以选择子集。"""
        x = {
            "f1": np.array([1.0, 2.0, 3.0]),
            "f2": np.array([4.0, 5.0, 6.0]),
            "f3": np.array([7.0, 8.0, 9.0]),
        }
        y = np.array([10.0, 11.0, 12.0])
        tool = StatisticsTool(x=x, y=y)

        # 只分析 f1 和 f3
        result = tool.execute(x_vars=["f1", "f3"])

        assert len(result["features"]) == 2
        feature_names = {f["name"] for f in result["features"]}
        assert feature_names == {"f1", "f3"}
