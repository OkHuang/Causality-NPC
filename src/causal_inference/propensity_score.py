"""
倾向性评分模块

功能：
- 计算倾向性评分（Propensity Score）
- 使用机器学习模型预测治疗概率
- 支持多种算法（逻辑回归、随机森林、梯度提升）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import logging

logger = logging.getLogger(__name__)


class PropensityScoreMatcher:
    """倾向性评分匹配器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化匹配器

        Parameters
        ----------
        config : dict
            配置字典
        """
        self.config = config or {}
        self.model = None
        self.fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        confounder_cols: List[str],
    ) -> "PropensityScoreMatcher":
        """
        拟合倾向性评分模型

        Parameters
        ----------
        df : pd.DataFrame
            数据
        treatment_col : str
            处理变量列名（0/1）
        confounder_cols : list
            混杂因素列名

        Returns
        -------
        self
        """
        logger.info("开始拟合倾向性评分模型...")

        # 准备数据
        X = df[confounder_cols].copy()
        y = df[treatment_col].copy()

        # 处理缺失值
        X = X.fillna(X.median())

        # 选择模型
        model_type = self.config.get("model", "logistic_regression")
        self.model = self._get_model(model_type)

        # 交叉验证
        if self.config.get("cross_validate", True):
            cv_scores = cross_val_score(
                self.model, X, y,
                cv=self.config.get("cv_folds", 5),
                scoring="roc_auc"
            )
            logger.info(f"交叉验证 AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # 拟合模型
        self.model.fit(X, y)

        # 校准（可选）
        if self.config.get("calibration", True):
            self.model = CalibratedClassifierCV(self.model, cv="prefit")
            self.model.fit(X, y)

        self.fitted = True
        self.confounder_cols = confounder_cols

        logger.info("倾向性评分模型拟合完成")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测倾向性评分

        Parameters
        ----------
        df : pd.DataFrame
            数据

        Returns
        -------
        np.ndarray
            倾向性评分（接受处理的概率）
        """
        if not self.fitted:
            raise ValueError("模型未拟合")

        X = df[self.confounder_cols].copy()
        X = X.fillna(X.median())

        propensities = self.model.predict_proba(X)[:, 1]

        return propensities

    def fit_transform(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        confounder_cols: List[str],
    ) -> pd.DataFrame:
        """
        拟合并转换（别名方法）

        这是 transform 方法的别名，提供一致的 API
        """
        return self.transform(df, treatment_col, confounder_cols)

    def transform(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        confounder_cols: List[str],
    ) -> pd.DataFrame:
        """
        拟合并转换，添加倾向性评分列

        Parameters
        ----------
        df : pd.DataFrame
            数据
        treatment_col : str
            处理变量列名
        confounder_cols : list
            混杂因素列名

        Returns
        -------
        pd.DataFrame
            添加了 propensity_score 列的数据
        """
        self.fit(df, treatment_col, confounder_cols)

        propensities = self.predict_proba(df)

        result = df.copy()
        result["propensity_score"] = propensities

        # 计算对数几率（logit）
        result["logit_ps"] = np.log(propensities / (1 - propensities + 1e-10))

        logger.info("已添加倾向性评分列")

        return result

    def _get_model(self, model_type: str):
        """
        根据配置返回模型
        """
        if model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, class_weight="balanced")
        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight="balanced",
                random_state=42,
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=42,
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def check_common_support(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        propensity_col: str = "propensity_score",
    ) -> Dict[str, Any]:
        """
        检查共同支持假设（Common Support）

        处理组和对照组的倾向性评分分布应该有重叠

        Parameters
        ----------
        df : pd.DataFrame
            包含倾向性评分的数据
        treatment_col : str
            处理变量列名
        propensity_col : str
            倾向性评分列名

        Returns
        -------
        dict
            共同支持检查结果
        """
        treatment_ps = df[df[treatment_col] == 1][propensity_col]
        control_ps = df[df[treatment_col] == 0][propensity_col]

        result = {
            "treatment_min": treatment_ps.min(),
            "treatment_max": treatment_ps.max(),
            "control_min": control_ps.min(),
            "control_max": control_ps.max(),
            "overlap_min": max(treatment_ps.min(), control_ps.min()),
            "overlap_max": min(treatment_ps.max(), control_ps.max()),
        }

        # 检查是否有足够的重叠
        if result["overlap_min"] >= result["overlap_max"]:
            logger.warning("警告：处理组和对照组的倾向性评分分布没有重叠！")
        else:
            logger.info(f"共同支持区间: [{result['overlap_min']:.3f}, {result['overlap_max']:.3f}]")

        return result
