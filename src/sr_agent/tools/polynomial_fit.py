# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""多项式拟合工具。

提供对输入数据的多项式拟合功能，支持自定义最高阶次数、交叉项控制等。
"""

import numpy as np
from itertools import combinations, product
from typing import Dict, Any, List, Optional, Tuple, Set

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register('polynomial_fit')
class PolynomialFitTool(BaseTool):
    """Fit polynomial models to data.

    This tool uses least squares regression to fit polynomials to input data,
    supporting:
    - Configurable maximum degree
    - Option to include interaction terms
    - Interaction blacklist (which variable pairs should not interact)
    - Interaction whitelist (only allow specified variable pairs to interact)

    Returned results include:
    - Fitted polynomial expression
    - Coefficient values and their statistical significance
    - Overall fit quality metrics (R², adjusted R², RMSE, etc.)

    Use cases:
    - Exploring nonlinear relationships between variables
    - Preprocessing or baseline for symbolic regression
    - Polynomial feature generation in feature engineering
    """

    metadata = ToolMetadata(
        name="polynomial_fit",
        description="Fit polynomial models to data with configurable degree and interaction terms. Returns fitted polynomials with statistical significance and fit quality metrics.",
        category="regression",
    )

    def execute(
        self,
        x_vars: Optional[List[str]] = None,
        y_var: str = "y",
        max_degree: int = 2,
        include_interactions: bool = True,
        interaction_blacklist: Optional[List[Tuple[str, str]]] = None,
        interaction_whitelist: Optional[List[Tuple[str, str]]] = None,
        include_bias: bool = True,
    ) -> Dict[str, Any]:
        """Execute polynomial fit.

        Args:
            x_vars: List of input feature names, e.g., ["x1", "x2"].
                None means use all features.
            y_var: Target variable name, default is "y".
            max_degree: Maximum polynomial degree, default is 2.
            include_interactions: Whether to include interaction terms, default is True.
            interaction_blacklist: List of variable pairs that should not interact.
                E.g., [("x1", "x2")] means no interaction between x1 and x2.
            interaction_whitelist: Only allow specified variable pairs to interact.
                If None, all pairs are allowed (unless in blacklist).
                If specified, only interactions in the whitelist are generated.
            include_bias: Whether to include bias/intercept term, default is True.

        Returns:
            Dictionary containing:
            - polynomial: Polynomial string expression
            - terms: Detailed information for each term
            - coefficients: Coefficient dictionary {term: coefficient}
            - fit_quality: Fit quality metrics (R², adjusted R², RMSE, AIC, BIC)
            - design_matrix_shape: Shape of the design matrix
            - n_parameters: Number of parameters
            - warnings: List of warning messages
        """
        x = self.context['x']
        y = self.context['y']

        # 选择要使用的变量
        if x_vars is None:
            x_vars = list(x.keys())

        # 从 context 中获取数据
        x = {var: x[var] for var in x_vars}

        warnings = []

        # 数据验证
        y = np.asarray(y).flatten()
        n_samples = len(y)

        var_names = list(x.keys())
        n_features = len(var_names)

        if n_features == 0:
            raise ValueError("至少需要一个输入特征")

        # 确保所有特征长度一致
        for name, arr in x.items():
            arr_checked = np.asarray(arr).flatten()
            if len(arr_checked) != n_samples:
                raise ValueError(f"特征 {name} 的长度 ({len(arr_checked)}) 与 y 的长度 ({n_samples}) 不一致")
            x[name] = arr_checked

        # 生成交叉项限制
        allowed_interactions = self._get_allowed_interactions(
            var_names,
            include_interactions,
            interaction_blacklist,
            interaction_whitelist
        )

        # 生成所有多项式项的幂次组合
        term_powers = self._generate_term_powers(
            var_names,
            max_degree,
            allowed_interactions
        )

        # 构建设计矩阵和项名称
        design_matrix, term_info = self._build_design_matrix(
            x, var_names, term_powers, include_bias
        )

        # 检查设计矩阵的秩
        matrix_rank = np.linalg.matrix_rank(design_matrix)
        n_params = design_matrix.shape[1]

        if matrix_rank < n_params:
            warnings.append(
                f"设计矩阵秩 deficient: 秩={matrix_rank}, 参数={n_params}。"
                "可能存在多重共线性，结果可能不稳定。"
            )

        # 使用最小二乘法拟合
        n = n_samples
        p = n_params

        try:
            # 使用 QR 分解提高数值稳定性
            Q, R = np.linalg.qr(design_matrix)
            coefficients = np.linalg.solve(R, Q.T @ y)

            # 计算残差
            y_pred = design_matrix @ coefficients
            residuals = y - y_pred

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
                    warnings.append("使用伪逆计算标准误差，结果可能不够精确。")
            else:
                std_errors = np.full(n_params, np.nan)
                warnings.append("样本数不足以计算标准误差。")

            # 计算 t 统计量和 p 值
            with np.errstate(divide='ignore', invalid='ignore'):
                t_stats = coefficients / std_errors
                # 使用 t 分布计算双尾 p 值
                from scipy import stats
                dof = max(n - p, 1)
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

        except Exception as e:
            # 降级到普通最小二乘
            warnings.append(f"QR 分解失败，使用普通最小二乘法：{str(e)}")
            coefficients, residuals, rank, s = np.linalg.lstsq(
                design_matrix, y, rcond=None
            )
            y_pred = design_matrix @ coefficients
            std_errors = np.full(n_params, np.nan)
            t_stats = np.full(n_params, np.nan)
            p_values = np.full(n_params, np.nan)

        # 计算拟合质量指标
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 调整 R²
        if n > p and ss_tot > 0:
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)
        else:
            adj_r_squared = np.nan

        # RMSE 和 MAE
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))

        # AIC 和 BIC
        if n > p and ss_res > 0:
            log_likelihood = -n/2 * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
            aic = 2 * p - 2 * log_likelihood
            bic = p * np.log(n) - 2 * log_likelihood
        else:
            aic = np.nan
            bic = np.nan

        # 构建结果
        terms_result = []
        polynomial_parts = []
        coefficients_dict = {}

        for i, (term_str, coef, std_err, t_stat, p_val) in enumerate(
            zip(term_info, coefficients, std_errors, t_stats, p_values)
        ):
            term_result = {
                "term": term_str,
                "coefficient": float(coef),
                "std_error": float(std_err) if not np.isnan(std_err) else None,
                "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
                "p_value": float(p_val) if not np.isnan(p_val) else None,
                "significant_at_0.05": bool(p_val < 0.05) if not np.isnan(p_val) else None,
                "significant_at_0.01": bool(p_val < 0.01) if not np.isnan(p_val) else None,
            }
            terms_result.append(term_result)
            coefficients_dict[term_str] = float(coef)

            # 构建多项式字符串部分
            if term_str == "1":
                polynomial_parts.insert(0, f"{coef:.6f}")
            elif coef != 0:
                if coef < 0:
                    sign = "-"
                    abs_coef = abs(coef)
                else:
                    sign = "+" if polynomial_parts else ""
                    abs_coef = coef

                if abs_coef == 1:
                    term_display = f"{sign} {term_str}"
                else:
                    term_display = f"{sign} {abs_coef:.6f}*{term_str}"
                polynomial_parts.append(term_display.strip())

        # 构建多项式字符串
        if not polynomial_parts:
            polynomial_str = "0"
        else:
            polynomial_str = " ".join(polynomial_parts)
            # 处理开头的 "+"
            if polynomial_str.startswith("+"):
                polynomial_str = polynomial_str[1:].strip()

        return {
            "polynomial": polynomial_str,
            "terms": terms_result,
            "coefficients": coefficients_dict,
            "fit_quality": {
                "r_squared": float(r_squared),
                "adjusted_r_squared": float(adj_r_squared) if not np.isnan(adj_r_squared) else None,
                "rmse": float(rmse),
                "mae": float(mae),
                "aic": float(aic) if not np.isnan(aic) else None,
                "bic": float(bic) if not np.isnan(bic) else None,
            },
            "residuals": residuals.tolist(),
            "predictions": y_pred.tolist(),
            "design_matrix_shape": design_matrix.shape,
            "n_parameters": n_params,
            "n_samples": n_samples,
            "matrix_rank": matrix_rank,
            "warnings": warnings,
        }

    def _get_allowed_interactions(
        self,
        var_names: List[str],
        include_interactions: bool,
        blacklist: Optional[List[Tuple[str, str]]],
        whitelist: Optional[List[Tuple[str, str]]],
    ) -> Set[Tuple[str, str]]:
        """Get allowed interaction term combinations.

        Args:
            var_names: List of variable names.
            include_interactions: Whether to include interaction terms.
            blacklist: List of variable pairs to exclude from interactions.
            whitelist: List of variable pairs to allow for interactions.

        Returns:
            Set of allowed variable pair combinations.
        """
        if not include_interactions:
            return set()

        # 生成所有可能的变量对
        all_pairs = set(combinations(sorted(var_names), 2))

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

    def _generate_term_powers(
        self,
        var_names: List[str],
        max_degree: int,
        allowed_interactions: Set[Tuple[str, str]],
    ) -> List[Tuple[int, ...]]:
        """Generate all polynomial term power combinations.

        Generates all power combinations satisfying total degree <= max_degree.

        Args:
            var_names: List of variable names.
            max_degree: Maximum polynomial degree.
            allowed_interactions: Set of allowed variable pair combinations.

        Returns:
            List of power combinations, each as a tuple representing
            the power of each variable.
        """
        n_vars = len(var_names)
        term_powers = []

        # 生成所有可能的幂次组合（使用 product 生成笛卡尔积）
        for powers in product(range(max_degree + 1), repeat=n_vars):
            total_degree = sum(powers)
            if total_degree == 0:
                # 截距项，总是添加
                term_powers.append(powers)
            elif total_degree <= max_degree:
                # 检查是否满足交叉项约束
                if self._check_interaction_constraint(
                    powers, var_names, allowed_interactions
                ):
                    term_powers.append(powers)

        # 按总次数排序，确保截距项在第一
        term_powers.sort(key=lambda p: (sum(p), p))

        return term_powers

    def _check_interaction_constraint(
        self,
        powers: Tuple[int, ...],
        var_names: List[str],
        allowed_interactions: Set[Tuple[str, str]],
    ) -> bool:
        """Check if power combination satisfies interaction constraints.

        Args:
            powers: Tuple of powers for each variable.
            var_names: List of variable names.
            allowed_interactions: Set of allowed variable pair combinations.

        Returns:
            True if constraints are satisfied, False otherwise.
        """
        if not allowed_interactions:
            # 不允许任何交叉项，检查是否有多于一个变量非零
            non_zero_count = sum(1 for p in powers if p > 0)
            return non_zero_count <= 1

        # 找出所有同时非零的变量对
        non_zero_indices = [i for i, p in enumerate(powers) if p > 0]

        for i, j in combinations(non_zero_indices, 2):
            pair = tuple(sorted((var_names[i], var_names[j])))
            if pair not in allowed_interactions:
                return False

        return True

    def _build_design_matrix(
        self,
        x: Dict[str, np.ndarray],
        var_names: List[str],
        term_powers: List[Tuple[int, ...]],
        include_bias: bool,
    ) -> Tuple[np.ndarray, List[str]]:
        """Build design matrix.

        Args:
            x: Input feature dictionary.
            var_names: List of variable names.
            term_powers: List of power combinations.
            include_bias: Whether to include bias/intercept term.

        Returns:
            Design matrix and list of term names.
        """
        n_samples = len(next(iter(x.values())))
        term_info = []
        columns = []

        # 添加所有多项式项
        for powers in term_powers:
            # 跳过截距项（如果不需要）
            if all(p == 0 for p in powers):
                if not include_bias:
                    continue
                # 添加截距项
                term_info.append("1")
                columns.append(np.ones(n_samples))
                continue

            # 计算当前项的值
            term_values = np.ones(n_samples)
            for var_idx, power in enumerate(powers):
                if power > 0:
                    var_name = var_names[var_idx]
                    term_values *= x[var_name] ** power

            columns.append(term_values)
            term_info.append(self._powers_to_string(powers, var_names))

        design_matrix = np.column_stack(columns) if columns else np.zeros((n_samples, 0))
        return design_matrix, term_info

    def _powers_to_string(
        self,
        powers: Tuple[int, ...],
        var_names: List[str],
    ) -> str:
        """Convert power combination to string representation.

        Args:
            powers: Tuple of powers.
            var_names: List of variable names.

        Returns:
            String representation, e.g., "x1**2*x2".
        """
        parts = []
        for var_name, power in zip(var_names, powers):
            if power == 0:
                continue
            elif power == 1:
                parts.append(var_name)
            else:
                parts.append(f"{var_name}**{power}")

        return "*".join(parts) if parts else "1"
