//
//  FaceAntiSpoofing.h
//  MobileFaceNet-iOS
//
//  Created by 张建伟 on 2020/1/22.
//  Copyright © 2020 周文鹏. All rights reserved.
//

#import <UIKit/UIKit.h>

static float fas_threshold = 0.2f; // 设置一个阙值，大于这个值认为是攻击
static int laplacian_threshold = 500; // 图片清晰度判断阙值


NS_ASSUME_NONNULL_BEGIN

@interface FaceAntiSpoofing : NSObject

- (float)antiSpoofing:(UIImage *)image;
- (int)laplacian:(UIImage *)image;

@end

NS_ASSUME_NONNULL_END
