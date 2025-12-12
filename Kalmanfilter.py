import numpy as np


class KalmanFilter2D:
    """用于二维坐标的卡尔曼滤波器"""

    def __init__(self, process_variance, measurement_variance, initial_value=0, initial_p=1.0):
        """
        初始化卡尔曼滤波器

        Args:
            process_variance: 过程噪声方差（Q），越小越信任预测值
            measurement_variance: 测量噪声方差（R），越小越信任测量值
            initial_value: 初始值
            initial_p: 初始估计误差协方差
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.value = initial_value
        self.p = initial_p
        self.initialized = False

    def update(self, measured_value):
        """
        使用测量值更新滤波器

        Args:
            measured_value: 新的测量值

        Returns:
            滤波后的值
        """
        if not self.initialized:
            self.value = measured_value
            self.initialized = True
            return self.value

        # 预测步骤
        p_pred = self.p + self.process_variance

        # 卡尔曼增益
        k = p_pred / (p_pred + self.measurement_variance)

        # 更新步骤
        self.value = self.value + k * (measured_value - self.value)
        self.p = (1 - k) * p_pred

        return self.value


class HandLandmarkKalmanFilter:
    """为手部21个关键点的卡尔曼滤波器容器"""

    def __init__(self, num_landmarks=21, process_variance=0.01, measurement_variance=0.1):
        """
        初始化手部关键点卡尔曼滤波器

        Args:
            num_landmarks: 关键点数量（默认为21）
            process_variance: 过程噪声方差
            measurement_variance: 测量噪声方差
        """
        self.num_landmarks = num_landmarks
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # 为每个关键点的x、y坐标创建独立的卡尔曼滤波器
        self.filters_x = [
            KalmanFilter2D(process_variance, measurement_variance)
            for _ in range(num_landmarks)
        ]
        self.filters_y = [
            KalmanFilter2D(process_variance, measurement_variance)
            for _ in range(num_landmarks)
        ]

        self.last_landmarks = None

    def update(self, landmarks):
        """
        更新所有关键点的滤波值

        Args:
            landmarks: 手部关键点列表，每个点为[x, y, confidence]或[x, y]

        Returns:
            滤波后的关键点列表
        """
        if landmarks is None or len(landmarks) != self.num_landmarks:
            return landmarks

        filtered_landmarks = []

        for i, landmark in enumerate(landmarks):
            x, y = landmark[0], landmark[1]

            # 分别对x、y坐标进行卡尔曼滤波
            filtered_x = self.filters_x[i].update(x)
            filtered_y = self.filters_y[i].update(y)

            # 保留原始的置信度（如果存在）
            if len(landmark) > 2:
                filtered_landmarks.append([filtered_x, filtered_y, landmark[2]])
            else:
                filtered_landmarks.append([filtered_x, filtered_y])

        self.last_landmarks = filtered_landmarks
        return filtered_landmarks

    def reset(self):
        """重置所有滤波器（切换手或手不可见时调用）"""
        self.filters_x = [
            KalmanFilter2D(self.process_variance, self.measurement_variance)
            for _ in range(self.num_landmarks)
        ]
        self.filters_y = [
            KalmanFilter2D(self.process_variance, self.measurement_variance)
            for _ in range(self.num_landmarks)
        ]
        self.last_landmarks = None


class MultiHandKalmanFilter:
    """支持多只手的卡尔曼滤波器管理器"""

    def __init__(self, process_variance=0.01, measurement_variance=0.1):
        """
        初始化多手卡尔曼滤波器管理器

        Args:
            process_variance: 过程噪声方差
            measurement_variance: 测量噪声方差
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        # 使用字典存储左右手的滤波器
        self.hand_filters = {
            'Right': HandLandmarkKalmanFilter(21, process_variance, measurement_variance),
            'Left': HandLandmarkKalmanFilter(21, process_variance, measurement_variance)
        }

    def update(self, hands):
        """
        更新所有手部的关键点

        Args:
            hands: 手部检测结果列表，包含hand信息

        Returns:
            更新后的hands列表
        """
        if not hands:
            return hands

        for hand in hands:
            # 获取手的类型（左手或右手）
            hand_type = hand.get('type', 'Right')  # 默认为右手

            # 获取该类型手的滤波器
            hand_filter = self.hand_filters[hand_type]

            # 更新关键点
            if 'lmList' in hand:
                hand['lmList'] = hand_filter.update(hand['lmList'])

        return hands

    def reset_hand(self, hand_type):
        """重置指定手的滤波器"""
        if hand_type in self.hand_filters:
            self.hand_filters[hand_type].reset()

    def reset_all(self):
        """重置所有滤波器"""
        for hand_filter in self.hand_filters.values():
            hand_filter.reset()























