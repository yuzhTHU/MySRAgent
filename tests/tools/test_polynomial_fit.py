# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""多项式拟合工具的单元测试。"""

import numpy as np
import pytest

from sr_agent.tools.polynomial_fit import PolynomialFitTool


class TestPolynomialFitTool:
    """测试 PolynomialFitTool 的正确性。"""

    def setup_method(self):
        """每个测试方法前执行。"""
        self.x = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])}
        self.y = np.array([3.1, 4.9, 7.1, 8.9, 11.1, 13.0, 14.9, 17.1, 18.9, 21.0])
        self.tool = PolynomialFitTool(x=self.x, y=self.y)

    def test_linear_fit(self):
        """测试一元线性拟合。"""
        # y = 2*x + 1 + noise
        result = self.tool.execute(max_degree=1)

        # 检查返回结构
        assert "polynomial" in result
        assert "terms" in result
        assert "fit_quality" in result

        # 检查拟合质量
        assert result["fit_quality"]["r_squared"] > 0.99

        # 检查系数（截距约 1，斜率约 2）
        terms_dict = {t["term"]: t["coefficient"] for t in result["terms"]}
        assert "1" in terms_dict
        assert "x" in terms_dict
        assert abs(terms_dict["1"] - 1.0) < 0.2
        assert abs(terms_dict["x"] - 2.0) < 0.1

    def test_quadratic_fit(self):
        """测试二次多项式拟合。"""
        # y = x^2 + 2*x + 1
        x = {"x": np.linspace(-5, 5, 50)}
        y = x["x"] ** 2 + 2 * x["x"] + 1 + np.random.normal(0, 0.1, 50)
        tool = PolynomialFitTool(x=x, y=y)

        result = tool.execute(max_degree=2)

        assert result["fit_quality"]["r_squared"] > 0.95
        terms = [t["term"] for t in result["terms"]]
        assert "x**2" in terms
        assert "x" in terms
        assert "1" in terms

    def test_multivariate_polynomial(self):
        """测试多元多项式拟合。"""
        # y = x1 + x2 + x1*x2
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + 1.5 * x1 * x2 + np.random.normal(0, 0.1, n)

        x = {"x1": x1, "x2": x2}
        tool = PolynomialFitTool(x=x, y=y)
        result = tool.execute(max_degree=2, include_interactions=True)

        assert result["fit_quality"]["r_squared"] > 0.9
        terms = [t["term"] for t in result["terms"]]
        assert "x1" in terms
        assert "x2" in terms
        assert "x1*x2" in terms

    def test_no_interactions(self):
        """测试不包含交叉项的情况。"""
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        x = {"x1": x1, "x2": x2}
        tool = PolynomialFitTool(x=x, y=y)
        result = tool.execute(
            max_degree=2, include_interactions=False
        )

        # 交叉项不应该存在
        terms = [t["term"] for t in result["terms"]]
        assert "x1*x2" not in terms

    def test_interaction_blacklist(self):
        """测试交叉项黑名单。"""
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        # y 包含 x1*x3 交叉项，但不包含 x1*x2
        y = 2 * x1 + 3 * x2 + 1.5 * x1 * x3 + np.random.normal(0, 0.1, n)

        x = {"x1": x1, "x2": x2, "x3": x3}
        tool = PolynomialFitTool(x=x, y=y)
        # 将 x1 和 x2 加入黑名单
        result = tool.execute(
            max_degree=2,
            include_interactions=True,
            interaction_blacklist=[("x1", "x2")],
        )

        # x1*x2 不应该存在
        terms = [t["term"] for t in result["terms"]]
        assert "x1*x2" not in terms
        # 但 x1*x3 和 x2*x3 应该存在
        assert "x1*x3" in terms or "x3*x1" in terms

    def test_interaction_whitelist(self):
        """测试交叉项白名单。"""
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + 1.5 * x1 * x2 + np.random.normal(0, 0.1, n)

        x = {"x1": x1, "x2": x2, "x3": x3}
        tool = PolynomialFitTool(x=x, y=y)
        # 只允许 x1 和 x2 之间的交叉项
        result = tool.execute(
            max_degree=2,
            include_interactions=True,
            interaction_whitelist=[("x1", "x2")],
        )

        # 只有 x1*x2 应该存在
        terms = [t["term"] for t in result["terms"]]
        has_x1_x3 = "x1*x3" in terms
        has_x2_x3 = "x2*x3" in terms
        assert not has_x1_x3
        assert not has_x2_x3

    def test_no_bias(self):
        """测试不包含截距项。"""
        x = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        tool = PolynomialFitTool(x=x, y=y)

        result = tool.execute(max_degree=1, include_bias=False)

        # 截距项不应该存在
        terms = [t["term"] for t in result["terms"]]
        assert "1" not in terms

    def test_higher_degree(self):
        """测试高次多项式拟合。"""
        x = {"x": np.linspace(-3, 3, 100)}
        # y = x^3 - 2*x^2 + x - 1
        y = x["x"] ** 3 - 2 * x["x"] ** 2 + x["x"] - 1 + np.random.normal(0, 0.1, 100)
        tool = PolynomialFitTool(x=x, y=y)

        result = tool.execute(max_degree=3)

        assert result["fit_quality"]["r_squared"] > 0.9
        terms = [t["term"] for t in result["terms"]]
        assert "x**3" in terms
        assert "x**2" in terms
        assert "x" in terms
        assert "1" in terms

    def test_fit_quality_metrics(self):
        """测试拟合质量指标的计算。"""
        x = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])}
        y = np.array([2.1, 4.0, 5.9, 8.1, 10.0, 12.1, 13.9, 16.0, 18.1, 19.9])
        tool = PolynomialFitTool(x=x, y=y)

        result = tool.execute(max_degree=1)

        fit_quality = result["fit_quality"]
        assert "r_squared" in fit_quality
        assert "adjusted_r_squared" in fit_quality
        assert "rmse" in fit_quality
        assert "mae" in fit_quality
        assert "aic" in fit_quality
        assert "bic" in fit_quality

        # R² 应该在 0 到 1 之间
        assert 0 <= fit_quality["r_squared"] <= 1
        assert 0 <= fit_quality["adjusted_r_squared"] <= 1

        # RMSE 和 MAE 应该是非负的
        assert fit_quality["rmse"] >= 0
        assert fit_quality["mae"] >= 0

    def test_term_significance(self):
        """测试项的显著性检验。"""
        x = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])}
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        tool = PolynomialFitTool(x=x, y=y)

        result = tool.execute(max_degree=1)

        for term_info in result["terms"]:
            if term_info["term"] != "1":  # 非截距项
                # p 值应该非常小（高度显著）
                if term_info["p_value"] is not None:
                    assert term_info["p_value"] < 0.05

    def test_design_matrix_info(self):
        """测试设计矩阵信息的返回。"""
        x = {"x1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "x2": np.array([2.0, 4.0, 6.0, 8.0, 10.0])}
        y = np.array([3.0, 6.0, 9.0, 12.0, 15.0])
        tool = PolynomialFitTool(x=x, y=y)

        result = tool.execute(max_degree=2)

        # 设计矩阵信息已移除，改用 terms 和 residuals_summary
        assert "terms" in result
        assert "residuals_summary" in result
        assert len(result["terms"]) > 0

    def test_polynomial_string_format(self):
        """测试多项式字符串格式。"""
        x = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        tool = PolynomialFitTool(x=x, y=y)

        result = tool.execute(max_degree=1)

        assert "polynomial" in result
        assert isinstance(result["polynomial"], str)
        assert len(result["polynomial"]) > 0

    def test_predictions_and_residuals(self):
        """测试预测值和残差的返回。"""
        x = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])
        tool = PolynomialFitTool(x=x, y=y)

        result = tool.execute(max_degree=1)

        # 残差统计信息
        assert "residuals_summary" in result
        summary = result["residuals_summary"]
        assert "min" in summary
        assert "max" in summary
        assert "mean" in summary
        assert "std" in summary

        # 残差均值应该接近 0（最小二乘性质）
        assert abs(summary["mean"]) < 1e-6

    def test_metadata_exists(self):
        """Test that metadata exists."""
        assert self.tool.metadata is not None
        assert self.tool.metadata.name == "polynomial_fit"
        assert "polynomial" in self.tool.metadata.description.lower()

    def test_multicollinearity_warning(self):
        """测试多重共线性警告。"""
        # 创建高度相关的特征
        x = {
            "x1": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "x2": np.array([2.0, 4.0, 6.0, 8.0, 10.0]),  # x2 = 2*x1
        }
        y = np.array([3.0, 6.0, 9.0, 12.0, 15.0])
        tool = PolynomialFitTool(x=x, y=y)

        result = tool.execute(max_degree=2)

        # 应该有警告
        assert "warnings" in result
        # 检查是否有秩 deficient 警告
        has_rank_warning = any(
            "秩" in w or "rank" in w.lower() or "共线" in w
            for w in result["warnings"]
        )
        # 由于 x2 = 2*x1，应该有多重共线性警告
        assert has_rank_warning

    def test_coefficients_dict_completeness(self):
        """测试系数字典的完整性。"""
        x = {"x": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        tool = PolynomialFitTool(x=x, y=y)

        result = tool.execute(max_degree=2)

        # terms 列表中的所有项都应该有完整的统计信息
        for term_info in result["terms"]:
            assert "term" in term_info
            assert "coefficient" in term_info
            assert "std_error" in term_info
            assert "p_value" in term_info

    def test_x_vars_subset(self):
        """测试 x_vars 参数可以选择子集。"""
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        y = 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        x = {"x1": x1, "x2": x2, "x3": x3}
        tool = PolynomialFitTool(x=x, y=y)

        # 只使用 x1 和 x2
        result = tool.execute(x_vars=["x1", "x2"], max_degree=2)

        # x3 不应该出现在 terms 中
        terms = [t["term"] for t in result["terms"]]
        assert all("x3" not in term for term in terms)
