//
//  MobileFaceNet.h
//  MobileFaceNet-iOS
//
//  Created by 周文鹏 on 2019/10/30.
//  Copyright © 2019 周文鹏. All rights reserved.
//

#import <UIKit/UIKit.h>

static float mfn_threshold = 0.8f; // 设置一个阙值，大于这个值认为是同一个人

NS_ASSUME_NONNULL_BEGIN

@interface MobileFaceNet : NSObject

/**
 比较两张人脸图片
 */
- (float)compare:(UIImage *)image1 with:(UIImage *)image2;

@end

NS_ASSUME_NONNULL_END
