//
//  ViewController.m
//  MobileFaceNet-iOS
//
//  Created by 周文鹏 on 2019/10/30.
//  Copyright © 2019 周文鹏. All rights reserved.
//

#import "ViewController.h"
#import "MTCNN.h"
#import "FaceAntiSpoofing.h"
#import "MobileFaceNet.h"
#import "Tools.h"

@interface ViewController () {
    MobileFaceNet *_mfn;
    FaceAntiSpoofing *_fas;
    MTCNN *_mtcnn;
}

@property (weak, nonatomic) IBOutlet UIImageView *imageView1;
@property (weak, nonatomic) IBOutlet UIImageView *imageView2;
@property (weak, nonatomic) IBOutlet UIImageView *cropImageView1;
@property (weak, nonatomic) IBOutlet UIImageView *cropImageView2;
@property (weak, nonatomic) IBOutlet UILabel *resultLabel;

@property (strong, nonatomic) UIImage *image1;
@property (strong, nonatomic) UIImage *image2;
@property (strong, nonatomic) UIImage *cropImage1;
@property (strong, nonatomic) UIImage *cropImage2;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    _mtcnn = [[MTCNN alloc] init];
    _fas = [[FaceAntiSpoofing alloc] init];
    _mfn = [[MobileFaceNet alloc] init];
    
    self.image1 = [UIImage imageNamed:@"trump1.jpg"];
    self.image2 = [UIImage imageNamed:@"trump2.jpg"];
    [self.imageView1 setImage:self.image1];
    [self.imageView2 setImage:self.image2];
}

/**
 人脸检测并剪裁
 */
- (IBAction)faceCrop:(id)sender {
    NSArray<Box *> *boxes1 = [_mtcnn detectFaces:self.image1 minFaceSize:40];
    NSArray<Box *> *boxes2 = [_mtcnn detectFaces:self.image2 minFaceSize:40];
    Box *box1 = boxes1[0];
    Box *box2 = boxes2[0];
    [box1 toSquareShape];
    [box2 toSquareShape];
    self.cropImage1 = [Tools cropImage:self.image1 toRect:box1.transform2Rect];
    self.cropImage2 = [Tools cropImage:self.image2 toRect:box2.transform2Rect];
    [self.cropImageView1 setImage:self.cropImage1];
    [self.cropImageView2 setImage:self.cropImage2];
}

/**
 活体检测
 */
- (IBAction)antiSpoofing:(id)sender {
    float score1 = [_fas antiSpoofing:self.cropImage1];
    float score2 = [_fas antiSpoofing:self.cropImage2];
    // 这个得分大于0.2认为是攻击，该模型我会不断更新到github上
    self.resultLabel.text = [NSString stringWithFormat:@"活体检测得分：%f, %f", score1, score2];
    self.resultLabel.textColor = [UIColor blackColor];
}

/**
 人脸比对
 */
- (IBAction)faceCompare:(id)sender {
    float same = [_mfn compare:self.cropImage1 with:self.cropImage2];
    if (same > mfn_threshold) {
        self.resultLabel.text = [NSString stringWithFormat:@"比对结果：YES，%f", same];
        self.resultLabel.textColor = [UIColor greenColor];
    } else {
        self.resultLabel.text = [NSString stringWithFormat:@"比对结果：NO，%f", same];
        self.resultLabel.textColor = [UIColor redColor];
    }
}

/**
 将mtcnn landmarks 画到图片上
 */
- (UIImage *)drawImage:(UIImage *)image withBox:(Box *)box {
    UIGraphicsBeginImageContext(CGSizeMake(CGImageGetWidth(image.CGImage), CGImageGetHeight(image.CGImage)));
    CGContextRef contextRef = UIGraphicsGetCurrentContext();
    [image drawAtPoint:CGPointMake(0, 0)];
    CGContextSetLineWidth(contextRef, 4);
    CGContextSetFillColorWithColor(contextRef, [[UIColor redColor] CGColor]);
    CGContextSetStrokeColorWithColor(contextRef, [[UIColor redColor] CGColor]);
    CGContextStrokeRect(contextRef, [box transform2Rect]);
    for (int i = 0; i < box.landmark.count; i++) {
        CGPoint point = [box.landmark[i] CGPointValue];
        CGContextAddArc(contextRef, point.x, point.y, 1, 0, 2*M_PI, 0);
        CGContextDrawPath(contextRef, kCGPathStroke);
    }
    UIImage *imageDrawed = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return imageDrawed;
}

@end
