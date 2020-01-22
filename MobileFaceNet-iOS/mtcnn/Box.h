//
//  Box.h
//  MobileFaceNet-iOS
//
//  Created by 周文鹏 on 2020/1/16.
//  Copyright © 2020 周文鹏. All rights reserved.
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface Box : NSObject

@property (strong, nonatomic) NSMutableArray *box;
@property (assign, nonatomic) CGFloat score;
@property (strong, nonatomic) NSMutableArray *bbr;
@property (assign, nonatomic) BOOL deleted;
@property (strong, nonatomic) NSMutableArray *landmark;

- (int)left;

- (int)right;

- (int)width;

- (int)height;

// 转为rect
- (CGRect)transform2Rect;

// 面积
- (int)area;

// Bounding Box Regression
- (void)calibrate;

// 当前box转为正方形
- (void)toSquareShape;

// 防止边界溢出，并维持square大小
- (void)limitSquareW:(int)w H:(int)h;

// 坐标是否越界
- (BOOL)transboundW:(int)w H:(int)h;

@end

NS_ASSUME_NONNULL_END
