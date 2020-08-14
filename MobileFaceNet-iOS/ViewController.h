//
//  ViewController.h
//  MobileFaceNet-iOS
//
//  Created by 周文鹏 on 2019/10/30.
//  Copyright © 2019 周文鹏. All rights reserved.
//

#import <UIKit/UIKit.h>
#import "MTCNN.h"
#import "FaceAntiSpoofing.h"
#import "MobileFaceNet.h"

@interface ViewController : UIViewController

@property (strong, nonatomic) MobileFaceNet *mfn;
@property (strong, nonatomic) FaceAntiSpoofing *fas;
@property (strong, nonatomic) MTCNN *mtcnn;

@property (weak, nonatomic) IBOutlet UIImageView *faceView;
@property (weak, nonatomic) IBOutlet UILabel *resultLabel;

@property (strong, nonatomic) UIImage *inputImage;

- (void)setText:(int)time laplace:(int)laplace score:(float)score compare:(float)compare;

@end

