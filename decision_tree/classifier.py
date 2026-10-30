"""Classification tree skeleton."""
import numpy as np
from .base import BaseDecisionTree
from .node import Node


class DecisionTreeClassifier(BaseDecisionTree):
    """Tree variant for classification.

    Parameters:
        max_depth (int or None): See ``BaseDecisionTree``.
        min_samples_split (int): See ``BaseDecisionTree``.
        criterion (str): Supported values: 'gini', 'entropy'.

    Each hook below documents what it receives and must return; see
    ``BaseDecisionTree`` for the full contract.
    """

    _valid_criteria = ("gini",)  # 'entropy' se habilita en el paso 4

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        super().__init__(max_depth, min_samples_split, criterion)

    def _build_tree(self, X, y, depth):
        """Recursively grow the tree.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Current depth (root is 0).

        Returns:
            Node: Leaf (when a stopping rule triggers) or internal node with
            two subtrees built from the best split.
        """
        # criterios de paro -> devolver una hoja
        if (len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
                or self._impurity(y) == 0.0):          # nodo puro
            return Node(value=self._leaf_value(y))

        split = self._best_split(X, y)
        if split is None:                              # ningun corte reduce impureza
            return Node(value=self._leaf_value(y))

        feature_index, threshold = split
        left_mask = X[:, feature_index] <= threshold
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature_index=feature_index, threshold=threshold,
                    left=left, right=right)

    def _impurity(self, y):
        """Impurity of ``y`` per ``self.criterion``: Gini impurity 1 - sum(p_k^2) or entropy -sum(p_k * log2(p_k)), where p_k is the fraction of samples of class k.

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity (0 if pure).
        """
        y = np.asarray(y)
        if y.size == 0:
            return 0.0
        # p_k = fraccion de muestras de cada clase
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        if self.criterion == "gini":
            return float(1.0 - np.sum(probs ** 2))
        # 'entropy' se agrega en el paso 4
        raise ValueError(f"Criterio no soportado (aun): {self.criterion!r}")

    def _best_split(self, X, y):
        """Search all features/thresholds for the largest impurity decrease.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)``, or
            None if no split reduces impurity.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)

        best_gain = 0.0
        best_split = None
        for feature_index in range(n_features):
            column = X[:, feature_index]
            # umbrales candidatos: puntos medios entre valores unicos ordenados
            # (para una columna one-hot de 0/1, el unico umbral es 0.5)
            values = np.unique(column)
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = column <= threshold
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue
                # impureza ponderada de los dos hijos
                weighted = (n_left * self._impurity(y[left_mask])
                            + n_right * self._impurity(y[~left_mask])) / n_samples
                gain = parent_impurity - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, float(threshold))
        return best_split

    def _leaf_value(self, y):
        """Return the majority (most frequent) class of y.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The prediction for the leaf.
        """
        y = np.asarray(y)
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _score(self, y_true, y_pred):
        """Metric for ``eval``: accuracy = fraction of correct predictions.

        Args:
            y_true (ndarray, shape (n_samples,)): True targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(y_true == y_pred))
