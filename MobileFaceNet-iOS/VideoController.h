//
//  VideoController.h
//  MobileFaceNet-iOS
//
//  Created by 张建伟 on 2020/8/14.
//  Copyright © 2020 周文鹏. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "ViewController.h"

NS_ASSUME_NONNULL_BEGIN

@interface VideoController : UIViewController

@property (assign, nonatomic) int type;
@property (assign, nonatomic) ViewController *viewController;

@end

NS_ASSUME_NONNULL_END
