//
//  MTCNN.h
//  MobileFaceNet-iOS
//
//  Created by 张建伟 on 2020/1/16.
//  Copyright © 2020 周文鹏. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "Box.h"

NS_ASSUME_NONNULL_BEGIN

@interface MTCNN : NSObject

/**
 检测人脸
 */
- (NSArray<Box *> *)detectFaces:(UIImage *)image minFaceSize:(int)minFaceSize;

@end

NS_ASSUME_NONNULL_END
