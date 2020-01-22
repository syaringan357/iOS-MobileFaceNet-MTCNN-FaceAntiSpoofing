//
//  Box.m
//  MobileFaceNet-iOS
//
//  Created by 周文鹏 on 2020/1/16.
//  Copyright © 2020 周文鹏. All rights reserved.
//

#import "Box.h"

@implementation Box

- (instancetype)init {
    if (self = [super init]) {
        self.box = [NSMutableArray array];
        self.bbr = [NSMutableArray array];
        self.deleted = NO;
        self.landmark = [NSMutableArray array];
    }
    return self;
}

- (int)left {
    return [self.box[0] intValue];
}

- (int)right {
    return [self.box[1] intValue];
}

- (int)width {
    return [self.box[2] intValue] - [self left] + 1;
}

- (int)height {
    return [self.box[3] intValue] - [self right] + 1;
}

- (CGRect)transform2Rect {
    CGRect rect = CGRectMake([self left], [self right], [self width], [self height]);
    return rect;
}

- (int)area {
    return [self width] * [self height];
}

- (void)calibrate {
    self.box[0] = @((int) ([self.box[0] intValue] + [self width] * [self.bbr[0] floatValue]));
    self.box[1] = @((int) ([self.box[1] intValue] + [self height] * [self.bbr[1] floatValue]));
    self.box[2] = @((int) ([self.box[2] intValue] + [self width] * [self.bbr[2] floatValue]));
    self.box[3] = @((int) ([self.box[3] intValue] + [self height] * [self.bbr[3] floatValue]));
    for (int i = 0; i < 4; i++) {
        self.bbr[i] = @(0.0f);
    }
}

- (void)toSquareShape {
    int w = [self width];
    int h = [self height];
    if (w > h) {
        self.box[1] = @([self.box[1] intValue] - (w - h) / 2);
        self.box[3] = @([self.box[3] intValue] + (w - h + 1) / 2);
    } else {
        self.box[0] = @([self.box[0] intValue] - (h - w) / 2);
        self.box[2] = @([self.box[2] intValue] + (h - w + 1) / 2);
    }
}

- (void)limitSquareW:(int)w H:(int)h {
    if ([self.box[0] intValue] < 0 || [self.box[1] intValue] < 0) {
        int len = MAX(-[self.box[0] intValue], -[self.box[1] intValue]);
        self.box[0] = @([self.box[0] intValue] + len);
        self.box[1] = @([self.box[1] intValue] + len);
    }
    if ([self.box[2] intValue] >= w || [self.box[3] intValue] >= h) {
        int len = MAX([self.box[2] intValue] - w + 1, [self.box[3] intValue] - h + 1);
        self.box[2] = @([self.box[2] intValue] - len);
        self.box[3] = @([self.box[3] intValue] - len);
    }
}

- (BOOL)transboundW:(int)w H:(int)h {
    if ([self.box[0] intValue] < 0 || [self.box[1] intValue] < 0) {
        return YES;
    } else if ([self.box[2] intValue] >= w || [self.box[3] intValue] >= h) {
        return YES;
    }
    return NO;
}

@end
