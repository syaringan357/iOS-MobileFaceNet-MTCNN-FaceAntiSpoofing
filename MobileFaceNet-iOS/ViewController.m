//
//  ViewController.m
//  MobileFaceNet-iOS
//
//  Created by 周文鹏 on 2019/10/30.
//  Copyright © 2019 周文鹏. All rights reserved.
//

#import "ViewController.h"
#import "Tools.h"
#import "VideoController.h"


@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.mtcnn = [[MTCNN alloc] init];
    self.fas = [[FaceAntiSpoofing alloc] init];
    self.mfn = [[MobileFaceNet alloc] init];
}

- (IBAction)inputFace:(id)sender {
    UIStoryboard *board = [UIStoryboard storyboardWithName:@"Main" bundle:nil];
    VideoController *controller = [board instantiateViewControllerWithIdentifier:@"VideoController"];
    controller.type = 1;
    controller.viewController = self;
    [self presentViewController:controller animated:YES completion:^{
        
    }];
}

- (IBAction)compareFace:(id)sender {
    UIStoryboard *board = [UIStoryboard storyboardWithName:@"Main" bundle:nil];
    VideoController *controller = [board instantiateViewControllerWithIdentifier:@"VideoController"];
    controller.type = 2;
    controller.viewController = self;
    [self presentViewController:controller animated:YES completion:^{
        
    }];
}

- (void)setText:(int)time laplace:(int)laplace score:(float)score compare:(float)compare {
    NSString *text = [NSString stringWithFormat:@"识别次数：%d\n清晰度：%d\n活体检测：%f", time, laplace, score];
    if (compare > 0) {
        text = [text stringByAppendingFormat:@"\n是同一人：%.2f%%", compare * 100];
    }
    self.resultLabel.text = text;
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
