# -*- coding: utf-8 -*-
"""
视频抽帧 + 全景拼接
从视频中抽取指定帧数的图像，然后拼接为全景图
完整独立实现，不依赖其他模块
支持透明背景
"""

import cv2
import numpy as np
import os
import time


class MultiImageStitcher:
    """多图像全景拼接器（支持透明背景）"""
    
    def __init__(self, 
                 ratio=0.75,
                 reproj_thresh=3.0,
                 ransac_iterations=2000,
                 use_weighted_blend=True,
                 blend_width=50,
                 use_transparent_bg=True):  # 新增参数
        """
        初始化拼接器
        
        参数:
            ratio: Lowe's Ratio Test 阈值
            reproj_thresh: RANSAC 重投影阈值
            ransac_iterations: RANSAC 迭代次数
            use_weighted_blend: 是否使用加权融合
            blend_width: 融合区域宽度
            use_transparent_bg: 是否使用透明背景（True=透明，False=黑色）
        """
        self.ratio = ratio
        self.reproj_thresh = reproj_thresh
        self.ransac_iterations = ransac_iterations
        self.use_weighted_blend = use_weighted_blend
        self.blend_width = blend_width
        self.use_transparent_bg = use_transparent_bg
    
    def convert_to_rgba(self, image):
        """将BGR图像转换为BGRA（添加Alpha通道）"""
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        return image
    
    def detect_and_describe(self, image):
        """检测关键点并提取 SIFT 特征描述符（使用灰度图）"""
        # 如果是RGBA，转换为灰度时忽略Alpha通道
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, descriptors1, descriptors2):
        """使用 FLANN 进行特征匹配，并应用 Lowe's Ratio Test 筛选"""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) >= 2:
                m, n = match_pair[0], match_pair[1]
                if m.distance < self.ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def compute_homography(self, keypoints1, keypoints2, good_matches):
        """使用 RANSAC 计算单应性矩阵"""
        if len(good_matches) < 4:
            return None, None, 0
            
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(
            points2, 
            points1, 
            cv2.RANSAC, 
            self.reproj_thresh,
            maxIters=self.ransac_iterations,
            confidence=0.995
        )
        
        inliers_count = int(np.sum(mask)) if mask is not None else 0
        
        return H, mask, inliers_count
    
    def warp_image_with_alpha(self, image, H, output_size):
        """
        对图像进行透视变换，并处理Alpha通道
        
        参数:
            image: 输入图像（BGR或BGRA）
            H: 单应性矩阵
            output_size: 输出图像尺寸 (width, height)
            
        返回:
            warped: 变换后的图像（BGRA格式）
        """
        # 确保图像是BGRA格式
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # 执行透视变换
        warped = cv2.warpPerspective(
            image, 
            H, 
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)  # 透明背景
        )
        
        return warped
    
    def stitch_two_images(self, image1, image2, image1_name="参考图像", image2_name="新图像"):
        """
        拼接两张图像，image2 拼接到 image1
        
        参数:
            image1: 参考图像（已拼接的全景图）
            image2: 新图像（要拼接上来的图像）
            image1_name: 图像 1 名称
            image2_name: 图像 2 名称
            
        返回:
            panorama: 拼接后的图像（BGRA格式，如果启用透明背景）
            info: 包含匹配信息的字典
        """
        start_total = time.time()
        
        # 转换为BGRA格式（如果需要透明背景）
        if self.use_transparent_bg:
            if image1.shape[2] == 3:
                image1 = self.convert_to_rgba(image1)
            if image2.shape[2] == 3:
                image2 = self.convert_to_rgba(image2)
        
        # Step 1: 特征检测（使用原始图像，忽略Alpha通道）
        start = time.time()
        keypoints1, descriptors1 = self.detect_and_describe(image1)
        keypoints2, descriptors2 = self.detect_and_describe(image2)
        time_detect = time.time() - start
        
        print(f"  特征检测：{time_detect:.3f}s")
        print(f"    - {image1_name} 关键点：{len(keypoints1)}")
        print(f"    - {image2_name} 关键点：{len(keypoints2)}")
        
        # Step 2: 特征匹配
        start = time.time()
        good_matches = self.match_features(descriptors1, descriptors2)
        time_match = time.time() - start
        
        print(f"  特征匹配：{time_match:.3f}s")
        print(f"    - 有效匹配点：{len(good_matches)}")
        
        if len(good_matches) < 4:
            print(f"  警告：匹配点不足，跳过此图像")
            return None, {'error': 'insufficient_matches', 'matches': len(good_matches)}
        
        # Step 3: 计算单应性矩阵
        start = time.time()
        H, mask, inliers_count = self.compute_homography(keypoints1, keypoints2, good_matches)
        time_homography = time.time() - start
        
        print(f"  单应性估计：{time_homography:.3f}s")
        print(f"    - 内点数量：{inliers_count}/{len(good_matches)} ({100*inliers_count/len(good_matches):.1f}%)")
        
        if H is None:
            print(f"  警告：无法计算单应性矩阵，跳过此图像")
            return None, {'error': 'homography_failed', 'matches': len(good_matches)}
        
        # Step 4: 图像变换与融合
        start = time.time()
        
        # 计算输出尺寸
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        # 变换 image2 的四个角到 image1 坐标系
        corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        
        # 计算全景图边界
        min_x = min(0, np.min(warped_corners[:, 0]))
        max_x = max(w1, np.max(warped_corners[:, 0]))
        min_y = min(0, np.min(warped_corners[:, 1]))
        max_y = max(h1, np.max(warped_corners[:, 1]))
        
        # 调整为正数
        output_width = int(max_x - min_x)
        output_height = int(max_y - min_y)
        
        # 创建平移矩阵，将负坐标移到正区域
        translation = np.array([[1, 0, -min_x],
                                [0, 1, -min_y],
                                [0, 0, 1]], dtype=np.float32)
        
        # 更新单应性矩阵
        H_updated = translation @ H
        
        # 变换图像（使用支持Alpha的变换函数）
        warped_image2 = self.warp_image_with_alpha(image2, H_updated, (output_width, output_height))
        warped_image1 = self.warp_image_with_alpha(image1, translation, (output_width, output_height))
        
        time_warp = time.time() - start
        
        # Step 5: 图像融合（支持Alpha通道）
        start = time.time()
        if self.use_weighted_blend:
            panorama = self.weighted_blend_with_alpha(warped_image1, warped_image2)
        else:
            panorama = self.alpha_composite(warped_image1, warped_image2)
        
        time_blend = time.time() - start
        
        total_time = time.time() - start_total
        print(f"  图像变换与融合：{time_warp + time_blend:.3f}s")
        print(f"  总耗时：{total_time:.3f}s")
        print(f"  全景图尺寸：{panorama.shape[1]}x{panorama.shape[0]}")
        
        info = {
            'keypoints1': len(keypoints1),
            'keypoints2': len(keypoints2),
            'matches': len(good_matches),
            'inliers': inliers_count,
            'homography': H,
            'total_time': total_time,
            'output_size': (panorama.shape[1], panorama.shape[0])
        }
        
        return panorama, info
    
    def alpha_composite(self, image1, image2):
        """
        使用Alpha通道合成两张图像
        
        参数:
            image1: 底图（BGRA）
            image2: 顶图（BGRA）
            
        返回:
            blended: 合成后的图像（BGRA）
        """
        # 分离通道
        b1, g1, r1, a1 = cv2.split(image1)
        b2, g2, r2, a2 = cv2.split(image2)
        
        # 转换为float类型便于计算
        a1 = a1.astype(np.float32) / 255.0
        a2 = a2.astype(np.float32) / 255.0
        
        # 计算输出Alpha通道
        out_alpha = a1 + a2 * (1 - a1)
        
        # 避免除零
        out_alpha = np.maximum(out_alpha, 1e-8)
        
        # 合成各通道
        out_b = (b1.astype(np.float32) * a1 + b2.astype(np.float32) * a2 * (1 - a1)) / out_alpha
        out_g = (g1.astype(np.float32) * a1 + g2.astype(np.float32) * a2 * (1 - a1)) / out_alpha
        out_r = (r1.astype(np.float32) * a1 + r2.astype(np.float32) * a2 * (1 - a1)) / out_alpha
        
        # 转换回uint8
        out_alpha = (out_alpha * 255).astype(np.uint8)
        out_b = np.clip(out_b, 0, 255).astype(np.uint8)
        out_g = np.clip(out_g, 0, 255).astype(np.uint8)
        out_r = np.clip(out_r, 0, 255).astype(np.uint8)
        
        # 合并通道
        blended = cv2.merge([out_b, out_g, out_r, out_alpha])
        
        return blended
    
    def weighted_blend_with_alpha(self, image1, image2):
        """
        带Alpha通道的加权融合
        
        参数:
            image1: 底图（BGRA）
            image2: 顶图（BGRA）
            
        返回:
            blended: 融合后的图像（BGRA）
        """
        # 获取Alpha通道作为权重
        a1 = image1[:, :, 3].astype(np.float32) / 255.0
        a2 = image2[:, :, 3].astype(np.float32) / 255.0
        
        # 计算总权重
        total_weight = a1 + a2
        total_weight = np.maximum(total_weight, 1e-8)
        
        # 归一化权重
        w1 = a1 / total_weight
        w2 = a2 / total_weight
        
        # 加权融合RGB通道
        blended_rgb = (image1[:, :, :3].astype(np.float32) * w1[:, :, np.newaxis] +
                      image2[:, :, :3].astype(np.float32) * w2[:, :, np.newaxis])
        
        # Alpha通道取最大值（或者也可以加权）
        blended_alpha = np.maximum(a1, a2) * 255
        
        # 合并结果
        blended = np.zeros_like(image1)
        blended[:, :, :3] = np.clip(blended_rgb, 0, 255).astype(np.uint8)
        blended[:, :, 3] = blended_alpha.astype(np.uint8)
        
        return blended
    
    def crop_transparent_edges(self, image):
        """
        裁剪透明边缘（基于Alpha通道）
        
        参数:
            image: BGRA图像
            
        返回:
            cropped: 裁剪后的图像
        """
        if image.shape[2] == 4:
            # 使用Alpha通道找到非透明区域
            alpha = image[:, :, 3]
            _, thresh = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(thresh)
            
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                cropped = image[y:y+h, x:x+w]
                return cropped
        return image
    
    def stitch_all_images(self, image_paths, output_path=None):
        """
        依次拼接所有图像
        
        参数:
            image_paths: 图像路径列表（按从左到右的顺序）
            output_path: 输出路径
            
        返回:
            panorama: 最终全景图（BGRA格式）
            all_info: 所有拼接步骤的信息
        """
        print(f"\n{'='*70}")
        print(f"开始多图像全景拼接")
        print(f"图像数量：{len(image_paths)}")
        print(f"透明背景：{'是' if self.use_transparent_bg else '否'}")
        print(f"{'='*70}\n")
        
        # 读取第一张图像作为基础
        print(f"加载第 1 张图像作为基础...")
        panorama = cv2.imdecode(np.fromfile(image_paths[0], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        
        if panorama is None:
            print(f"错误：无法读取图像 {image_paths[0]}")
            return None, []
        
        # 转换为BGRA格式
        if panorama.shape[2] == 3:
            panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2BGRA)
        
        print(f"基础图像尺寸：{panorama.shape[1]}x{panorama.shape[0]}\n")
        
        all_info = []
        failed_images = []
        
        # 依次拼接后续图像
        for i in range(1, len(image_paths)):
            print(f"\n{'-'*70}")
            print(f"拼接第 {i+1} 张图像：{os.path.basename(image_paths[i])}")
            print(f"{'-'*70}")
            
            # 读取新图像
            new_image = cv2.imdecode(np.fromfile(image_paths[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            
            if new_image is None:
                print(f"错误：无法读取图像 {image_paths[i]}")
                failed_images.append(image_paths[i])
                continue
            
            # 拼接
            result, info = self.stitch_two_images(
                panorama, 
                new_image,
                image1_name=f"当前全景图",
                image2_name=os.path.basename(image_paths[i])
            )
            
            if result is not None:
                panorama = result
                info['image'] = image_paths[i]
                all_info.append(info)
                print(f"✓ 拼接成功")
            else:
                print(f"✗ 拼接失败：{info.get('error', '未知错误')}")
                failed_images.append(image_paths[i])
        
        # 裁剪透明边缘
        print(f"\n{'-'*70}")
        print(f"裁剪透明边缘...")
        panorama_cropped = self.crop_transparent_edges(panorama)
        print(f"裁剪后尺寸：{panorama_cropped.shape[1]}x{panorama_cropped.shape[0]}")
        
        # 保存结果
        if output_path:
            print(f"\n保存全景图到：{output_path}")
            # 保存为PNG以保留Alpha通道
            output_path_png = output_path.replace('.jpg', '.png')
            cv2.imencode('.png', panorama)[1].tofile(output_path_png)
            
            output_path_cropped = output_path_png.replace('.png', '_cropped.png')
            cv2.imencode('.png', panorama_cropped)[1].tofile(output_path_cropped)
            
            # 同时保存一个JPG版本（黑色背景，供预览）
            if self.use_transparent_bg:
                # 创建黑色背景的JPG版本
                black_bg = np.zeros((panorama.shape[0], panorama.shape[1], 3), dtype=np.uint8)
                alpha = panorama[:, :, 3].astype(np.float32) / 255.0
                for c in range(3):
                    black_bg[:, :, c] = (panorama[:, :, c] * alpha).astype(np.uint8)
                output_path_jpg = output_path.replace('.jpg', '_preview.jpg')
                cv2.imencode('.jpg', black_bg)[1].tofile(output_path_jpg)
        
        # 打印总结
        print(f"\n{'='*70}")
        print(f"拼接完成总结")
        print(f"{'='*70}")
        print(f"成功拼接：{len(all_info)} 张图像")
        if failed_images:
            print(f"失败图像：{len(failed_images)} 张")
            for img in failed_images:
                print(f"  - {os.path.basename(img)}")
        print(f"最终全景图尺寸：{panorama_cropped.shape[1]}x{panorama_cropped.shape[0]}")
        if self.use_transparent_bg:
            print(f"注意：全景图已保存为PNG格式以保留透明背景")
            print(f"      JPG预览版本（黑色背景）也已保存")
        print(f"{'='*70}\n")
        
        return panorama_cropped, all_info


def extract_frames_from_video(video_path, output_dir, num_frames=10):
    """
    从视频中抽取指定数量的帧
    
    参数:
        video_path: 视频路径
        output_dir: 输出目录
        num_frames: 抽取的帧数（可修改）
        
    返回:
        frame_paths: 抽取的帧路径列表
        frame_numbers: 对应的帧序号
    """
    print(f"\n{'='*70}")
    print(f"视频抽帧")
    print(f"{'='*70}")
    print(f"视频路径：{video_path}")
    print(f"抽取帧数：{num_frames}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return [], []
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    
    print(f"视频总帧数：{total_frames}")
    print(f"帧率：{fps} fps")
    print(f"视频时长：{duration:.2f} 秒")
    
    if total_frames < num_frames:
        print(f"警告：视频总帧数 ({total_frames}) 少于要抽取的帧数 ({num_frames})")
        print(f"将抽取所有 {total_frames} 帧")
        num_frames = total_frames
    
    # 计算等间隔抽帧的索引
    # 从第 1 帧开始，到最后一帧结束，均匀分布
    if num_frames == 1:
        frame_indices = [total_frames // 2]  # 只抽 1 帧时取中间帧
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    print(f"\n抽取的帧序号：{frame_indices}")
    print(f"对应时间点：{[f'{i/fps:.2f}s' for i in frame_indices]}")
    
    # 抽取帧
    frame_paths = []
    frame_numbers = []
    
    start_time = time.time()
    
    for idx, frame_idx in enumerate(frame_indices):
        # 设置视频位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 读取帧
        ret, frame = cap.read()
        
        if not ret:
            print(f"警告：无法读取第 {frame_idx} 帧")
            continue
        
        # 保存帧
        frame_name = f'frame_{idx:03d}.jpg'
        frame_path = os.path.join(output_dir, frame_name)
        
        # 使用 imencode 处理中文路径
        cv2.imencode('.jpg', frame)[1].tofile(frame_path)
        
        frame_paths.append(frame_path)
        frame_numbers.append(frame_idx)
        
        print(f"  [{idx+1}/{num_frames}] 已保存：{frame_name} (帧序号：{frame_idx}, 时间：{frame_idx/fps:.2f}s)")
    
    cap.release()
    
    extract_time = time.time() - start_time
    print(f"\n抽帧完成，耗时：{extract_time:.2f}s")
    print(f"成功抽取：{len(frame_paths)} 帧")
    
    return frame_paths, frame_numbers


def stitch_frames(frame_paths, output_dir, use_transparent_bg=True):
    """
    拼接抽取的帧
    
    参数:
        frame_paths: 帧路径列表
        output_dir: 输出目录
        use_transparent_bg: 是否使用透明背景
        
    返回:
        panorama: 全景图
        info: 拼接信息
    """
    print(f"\n{'='*70}")
    print(f"全景拼接")
    print(f"{'='*70}")
    print(f"待拼接图像数量：{len(frame_paths)}")
    
    if len(frame_paths) < 2:
        print(f"错误：至少需要 2 张图像才能拼接")
        return None, None
    
    # 创建拼接器（启用透明背景）
    stitcher = MultiImageStitcher(
        ratio=0.75,              # Lowe's Ratio Test 阈值
        reproj_thresh=3.0,       # RANSAC 重投影阈值
        ransac_iterations=2000,  # RANSAC 迭代次数
        use_weighted_blend=True, # 启用加权融合
        blend_width=50,          # 融合区域宽度
        use_transparent_bg=use_transparent_bg  # 使用透明背景
    )
    
    # 执行拼接
    start_time = time.time()
    
    # 读取第一张图像作为基础
    print(f"\n加载第 1 张图像作为基础...")
    panorama = cv2.imdecode(np.fromfile(frame_paths[0], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    
    if panorama is None:
        print(f"错误：无法读取图像 {frame_paths[0]}")
        return None, None
    
    # 转换为BGRA格式
    if panorama.shape[2] == 3:
        panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2BGRA)
    
    print(f"基础图像尺寸：{panorama.shape[1]}x{panorama.shape[0]}\n")
    
    all_info = []
    failed_images = []
    
    # 依次拼接后续图像
    for i in range(1, len(frame_paths)):
        print(f"拼接第 {i+1} 张图像：{os.path.basename(frame_paths[i])}")
        
        # 读取新图像
        new_image = cv2.imdecode(np.fromfile(frame_paths[i], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        
        if new_image is None:
            print(f"  错误：无法读取图像 {frame_paths[i]}")
            failed_images.append(frame_paths[i])
            continue
        
        # 转换为BGRA格式
        if new_image.shape[2] == 3:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2BGRA)
        
        # 拼接
        result, info = stitcher.stitch_two_images(
            panorama, 
            new_image,
            image1_name=f"当前全景图",
            image2_name=os.path.basename(frame_paths[i])
        )
        
        if result is not None:
            panorama = result
            info['image'] = frame_paths[i]
            all_info.append(info)
            print(f"  ✓ 拼接成功")
        else:
            print(f"  ✗ 拼接失败：{info.get('error', '未知错误')}")
            failed_images.append(frame_paths[i])
    
    # 裁剪透明边缘
    print(f"\n裁剪透明边缘...")
    panorama_cropped = stitcher.crop_transparent_edges(panorama)
    print(f"裁剪后尺寸：{panorama_cropped.shape[1]}x{panorama_cropped.shape[0]}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    if use_transparent_bg:
        # 保存PNG格式（保留透明通道）
        output_path_png = os.path.join(output_dir, 'video_panorama.png')
        print(f"\n保存全景图（透明背景）到：{output_path_png}")
        cv2.imencode('.png', panorama)[1].tofile(output_path_png)
        
        output_path_cropped = os.path.join(output_dir, 'video_panorama_cropped.png')
        cv2.imencode('.png', panorama_cropped)[1].tofile(output_path_cropped)
        
        # 同时保存JPG预览（黑色背景）
        black_bg = np.zeros((panorama.shape[0], panorama.shape[1], 3), dtype=np.uint8)
        alpha = panorama[:, :, 3].astype(np.float32) / 255.0
        for c in range(3):
            black_bg[:, :, c] = (panorama[:, :, c] * alpha).astype(np.uint8)
        
        output_path_jpg = os.path.join(output_dir, 'video_panorama_preview.jpg')
        cv2.imencode('.jpg', black_bg)[1].tofile(output_path_jpg)
        
        # 也保存裁剪版的预览
        black_bg_cropped = black_bg[:panorama_cropped.shape[0], :panorama_cropped.shape[1]]
        output_path_cropped_jpg = os.path.join(output_dir, 'video_panorama_cropped_preview.jpg')
        cv2.imencode('.jpg', black_bg_cropped)[1].tofile(output_path_cropped_jpg)
    else:
        # 保存JPG格式（黑色背景）
        output_path = os.path.join(output_dir, 'video_panorama.jpg')
        print(f"\n保存全景图到：{output_path}")
        cv2.imencode('.jpg', panorama)[1].tofile(output_path)
        
        output_path_cropped = output_path.replace('.jpg', '_cropped.jpg')
        cv2.imencode('.jpg', panorama_cropped)[1].tofile(output_path_cropped)
    
    # 打印总结
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"拼接完成总结")
    print(f"{'='*70}")
    print(f"成功拼接：{len(all_info)} 张图像")
    if failed_images:
        print(f"失败图像：{len(failed_images)} 张")
        for img in failed_images:
            print(f"  - {os.path.basename(img)}")
    print(f"最终全景图尺寸：{panorama_cropped.shape[1]}x{panorama_cropped.shape[0]}")
    print(f"总耗时：{total_time:.2f}s")
    print(f"{'='*70}\n")
    
    return panorama_cropped, {
        'all_info': all_info,
        'failed_images': failed_images,
        'total_time': total_time,
        'output_size': (panorama_cropped.shape[1], panorama_cropped.shape[0])
    }


def main():
    """主函数"""
    
    # ========== 配置区域（可修改） ==========
    
    # 抽取的帧数（可修改）
    NUM_FRAMES = 4  # <-- 修改这里来改变抽取的帧数
    
    # 是否使用透明背景（True=透明背景，False=黑色背景）
    USE_TRANSPARENT_BG = True  # <-- 修改这里来控制背景是否透明
    
    # 输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frames_output_dir = os.path.join(script_dir, 'extracted_frames')
    panorama_output_dir = os.path.join(script_dir, 'output')
    
    # 视频路径（相对路径）
    video_path = os.path.join(script_dir, 'video', 'IMG_2376.MOV')
    
    # =====================================
    
    print(f"\n{'#'*70}")
    print(f"# 视频抽帧 + 全景拼接 - 透明背景版")
    print(f"{'#'*70}")
    
    # Step 1: 抽帧
    frame_paths, frame_numbers = extract_frames_from_video(
        video_path, 
        frames_output_dir, 
        num_frames=NUM_FRAMES
    )
    
    if not frame_paths:
        print(f"\n错误：抽帧失败")
        return
    
    # Step 2: 拼接（使用透明背景）
    panorama, info = stitch_frames(frame_paths, panorama_output_dir, use_transparent_bg=USE_TRANSPARENT_BG)
    
    if panorama is not None:
        print(f"\n✓ 全景图已成功保存:")
        if USE_TRANSPARENT_BG:
            print(f"  - 透明背景版：{os.path.join(panorama_output_dir, 'video_panorama.png')}")
            print(f"  - 透明背景裁剪版：{os.path.join(panorama_output_dir, 'video_panorama_cropped.png')}")
            print(f"  - JPG预览版（黑色背景）：{os.path.join(panorama_output_dir, 'video_panorama_preview.jpg')}")
            print(f"  - JPG预览裁剪版：{os.path.join(panorama_output_dir, 'video_panorama_cropped_preview.jpg')}")
        else:
            print(f"  - 完整版：{os.path.join(panorama_output_dir, 'video_panorama.jpg')}")
            print(f"  - 裁剪版：{os.path.join(panorama_output_dir, 'video_panorama_cropped.jpg')}")
        print(f"  - 抽取的帧目录：{frames_output_dir}")
    else:
        print(f"\n✗ 拼接失败")


if __name__ == '__main__':
    main()