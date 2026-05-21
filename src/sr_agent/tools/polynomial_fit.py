# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""多项式拟合工具。提供对输入变量或表达式的多项式拟合功能，支持自定义最高阶次数、交叉项控制等。"""

import numpy as np
import nd2py as nd
from itertools import combinations, product
from functools import reduce
from typing import Dict, Any, List, Optional, Tuple, Set
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('polynomial_fit')
class PolynomialFitTool(BaseTool):
    metadata = ToolMetadata(name="polynomial_fit")

    def execute(
        self,
        x: List[str] = None,
        y: str = None,
        max_degree: int = 2,
        include_interactions: bool = True,
        interaction_blacklist: List[Tuple[str, str]] = None,
        interaction_whitelist: List[Tuple[str, str]] = None,
        include_bias: bool = True,
    ) -> Dict[str, Any]:
        """Execute polynomial fit.

        Args:
            x: List of input feature names, e.g., ["x1", "x2"]. Use all features other than y by default.
                Expressions are also supported, e.g., ["sin(x1)", "(x1-x2)**2"].
            y: Target variable name. Use target variable by default.
                Expressions are also supported, e.g., "log(y)", "y - x1"
            max_degree: Maximum polynomial degree.
            include_interactions: Whether to include interaction terms.
            interaction_blacklist: List of variable pairs that should not interact.
                E.g., [("x1", "x2")] means no interaction between x1 and x2.
            interaction_whitelist: Only allow specified variable pairs to interact.
                By default, all pairs are allowed (unless in blacklist).
                If specified, only interactions in the whitelist are generated.
            include_bias: Whether to include bias/intercept term.
        """
        data = self.context["data"]
        y = y or self.context["target"]
        y = y.strip().strip('"').strip("'")
        x = x or [var for var in data if var != y]
        exceptions = []

        try:
            eq_y = nd.parse(y.replace('^', '**').replace('np.', ''))
            data_y = eq_y.eval(data).flatten()
        except Exception as e:
            raise ValueError(
                f"Failed to compute target '{y}': {str(e)}" +
                "\nOther exceptions: " + "; ".join(exceptions)
            )

        eq_x_list = []
        for xi in x:
            try:
                eq_x = nd.parse(xi.replace('^', '**').replace('np.', ''))
                data_x = eq_x.eval(data).flatten()
                eq_x_list.append(eq_x)
                assert data_x.shape == data_y.shape, f"Feature '{xi}' shape {data_x.shape} does not match target shape {data_y.shape}."
            except Exception as e:
                exceptions.append(f"Failed to compute feature '{xi}': {str(e)}")
        if len(eq_x_list) == 0:
            raise ValueError(
                "No valid input variables available for fitting.\n" +
                "Other exceptions: " + "; ".join(exceptions)
            )

        # 生成交叉项限制
        allowed_interactions = self._get_allowed_interactions(
            eq_x_list, include_interactions, interaction_blacklist, interaction_whitelist
        )

        # 构建总次数不超过 max_degree 的符号项，并统一计算设计矩阵
        terms = self.generate_terms(eq_x_list, max_degree, allowed_interactions, include_bias)
        design_matrix = self._build_design_matrix(data, terms, len(data_y))

        # 检查设计矩阵的秩
        matrix_rank = np.linalg.matrix_rank(design_matrix)
        n_params = design_matrix.shape[1]

        if matrix_rank < n_params:
            exceptions.append(
                f"设计矩阵秩 deficient: 秩={matrix_rank}, 参数={n_params}。"
                "可能存在多重共线性，结果可能不稳定。"
            )

        # 使用最小二乘法拟合
        p = n_params
        n = len(data_y)

        try:
            # 使用 QR 分解提高数值稳定性
            Q, R = np.linalg.qr(design_matrix)
            coefficients = np.linalg.solve(R, Q.T @ data_y)

            # 计算残差
            y_pred = design_matrix @ coefficients
            residuals = data_y - y_pred

            # 计算系数标准误差
            if n > p:
                mse = np.sum(residuals ** 2) / (n - p)
                # 系数的协方差矩阵
                try:
                    cov_matrix = mse * np.linalg.inv(R.T @ R)
                    std_errors = np.sqrt(np.diag(cov_matrix))
                except np.linalg.LinAlgError:
                    # 如果矩阵奇异，使用伪逆
                    cov_matrix = mse * np.linalg.pinv(R.T @ R)
                    std_errors = np.sqrt(np.diag(cov_matrix))
                    exceptions.append("使用伪逆计算标准误差，结果可能不够精确。")
            else:
                std_errors = np.full(n_params, np.nan)
                exceptions.append("样本数不足以计算标准误差。")

            # 计算 t 统计量和 p 值
            with np.errstate(divide='ignore', invalid='ignore'):
                t_stats = coefficients / std_errors
                # 使用 t 分布计算双尾 p 值
                from scipy import stats
                dof = max(n - p, 1)
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

        except Exception as e:
            # 降级到普通最小二乘
            exceptions.append(f"QR 分解失败，使用普通最小二乘法：{str(e)}")
            coefficients, residuals, rank, s = np.linalg.lstsq(
                design_matrix, data_y, rcond=None
            )
            y_pred = design_matrix @ coefficients
            std_errors = np.full(n_params, np.nan)
            t_stats = np.full(n_params, np.nan)
            p_values = np.full(n_params, np.nan)

        # 计算拟合质量指标
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((data_y - np.mean(data_y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else np.nan

        # AIC 和 BIC
        if n > p and ss_res > 0:
            log_likelihood = -n/2 * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
            aic = 2 * p - 2 * log_likelihood
            bic = p * np.log(n) - 2 * log_likelihood
        else:
            aic = np.nan
            bic = np.nan

        # 构建多项式
        polynomial_parts = [float(coef) * term for coef, term in zip(coefficients, terms) if coef != 0]
        polynomial = reduce(lambda a, b: a + b, polynomial_parts) if polynomial_parts else nd.parse("0")

        # terms_result = []
        # for term, coef, std_err, t_stat, p_val in zip(terms, coefficients, std_errors, t_stats, p_values):
        #     terms_result.append({
        #         "term": term.to_str(),
        #         "coefficient": float(coef),
        #         "std_error": float(std_err) if not np.isnan(std_err) else None,
        #         "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
        #         "p_value": float(p_val) if not np.isnan(p_val) else None,
        #         "significant_at_0.05": bool(p_val < 0.05) if not np.isnan(p_val) else None,
        #         "significant_at_0.01": bool(p_val < 0.01) if not np.isnan(p_val) else None,
        #     })

        # 检查是否为有效的候选目标公式
        is_candidate = (y == self.context['target']) and (y not in x)

        results = {
            "formula": polynomial.to_str(),
            "metrics": self.evaluate(y_pred=y_pred, y_true=data_y) | {
                "adjusted_r2": adjusted_r2, "aic": aic, "bic": bic,
            },
            # "terms": terms_result,
            "is_candidate": is_candidate,
            "exceptions": exceptions,
        }
        return results

    def _get_allowed_interactions(
        self,
        features: List[nd.Symbol],
        include_interactions: bool,
        blacklist: Optional[List[Tuple[str, str]]],
        whitelist: Optional[List[Tuple[str, str]]],
    ) -> Set[Tuple[str, str]]:
        """Get allowed interaction term combinations.

        Args:
            features: List of symbolic features.
            include_interactions: Whether to include interaction terms.
            blacklist: List of variable pairs to exclude from interactions.
            whitelist: List of variable pairs to allow for interactions.

        Returns:
            Set of allowed variable pair combinations.
        """
        if not include_interactions:
            return set()

        # 生成所有可能的变量对
        all_pairs = set(combinations(sorted([f.to_str() for f in features]), 2))

        if whitelist is not None:
            # 白名单模式：只允许白名单中的组合
            whitelist_normalized = set(
                tuple(sorted(pair)) for pair in whitelist
            )
            allowed = all_pairs & whitelist_normalized
        else:
            # 默认允许所有组合，除非在黑名单中
            allowed = all_pairs

        if blacklist is not None:
            blacklist_normalized = set(
                tuple(sorted(pair)) for pair in blacklist
            )
            allowed -= blacklist_normalized

        return allowed

    def generate_terms(
        self,
        features: List[nd.Symbol],
        max_degree: int,
        allowed_interactions: Set[Tuple[str, str]],
        include_bias: bool,
    ) -> List[nd.Symbol]:
        """Generate symbolic terms whose total degree is no more than max_degree."""
        n_vars = len(features)
        terms = []
        for powers in sorted(product(range(max_degree + 1), repeat=n_vars), key=lambda p: (sum(p), p)):
            total_degree = sum(powers)
            if total_degree == 0:
                if not include_bias:
                    continue
                terms.append(nd.parse("1"))
                continue
            if total_degree > max_degree:
                continue

            non_zero_indices = [i for i, power in enumerate(powers) if power > 0]
            if not allowed_interactions and len(non_zero_indices) > 1:
                continue
            if allowed_interactions:
                allowed = True
                for i, j in combinations(non_zero_indices, 2):
                    pair = tuple(sorted((features[i].to_str(), features[j].to_str())))
                    if pair not in allowed_interactions:
                        allowed = False
                        break
                if not allowed:
                    continue

            factors = []
            for feature, power in zip(features, powers):
                if power > 0:
                    factors.append(feature if power == 1 else feature ** power)
            terms.append(reduce(lambda a, b: a * b, factors))
        return terms

    def _build_design_matrix(
        self,
        data: Dict[str, np.ndarray],
        terms: List[nd.Symbol],
        n_samples: int,
    ) -> np.ndarray:
        """Evaluate symbolic terms to build the design matrix."""
        columns = []
        for term in terms:
            values = np.asarray(term.eval(data))
            if values.ndim == 0:
                values = np.full(n_samples, float(values))
            else:
                values = values.flatten()
            columns.append(values)
        return np.column_stack(columns) if columns else np.zeros((n_samples, 0))
