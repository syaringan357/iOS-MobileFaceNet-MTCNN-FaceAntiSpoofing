//
//  Tools.h
//  MobileFaceNet-iOS
//
//  Created by 周文鹏 on 2020/1/15.
//  Copyright © 2020 周文鹏. All rights reserved.
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface Tools : NSObject

// 获取文件路径
+ (NSString *)filePathForResourceName:(NSString *)name extension:(NSString *)extension;

// UIImage和uint8_t互转
+ (UInt8 *)convertUIImageToBitmapRGBA8:(UIImage *)image;
+ (UIImage *)convertBitmapRGBA8ToUIImage:(uint8_t *)buffer withWidth:(int)width withHeight:(int)height;

// 缩放图片
+ (UIImage *)scaleImage:(UIImage *)image toSize:(CGSize)size;
+ (UIImage *)scaleImage:(UIImage *)image toScale:(float)scale;

// 裁剪图片
+ (UIImage *)cropImage:(UIImage *)image toRect:(CGRect)rect;

@end

NS_ASSUME_NONNULL_END
