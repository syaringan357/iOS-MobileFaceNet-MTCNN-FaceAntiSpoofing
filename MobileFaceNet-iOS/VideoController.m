//
//  VideoController.m
//  MobileFaceNet-iOS
//
//  Created by 张建伟 on 2020/8/14.
//  Copyright © 2020 周文鹏. All rights reserved.
//

#import "VideoController.h"
#import <AVFoundation/AVFoundation.h>
#import "Tools.h"
#import "FaceAntiSpoofing.h"
#import "MobileFaceNet.h"

@interface VideoController () <AVCaptureVideoDataOutputSampleBufferDelegate>

@property (weak, nonatomic) IBOutlet UIView *preview;
@property (weak, nonatomic) IBOutlet UILabel *resultLabel;

@property (strong, nonatomic) AVCaptureSession *session;
@property (strong, nonatomic) AVCaptureDeviceInput *input;
@property (strong, nonatomic) AVCaptureVideoDataOutput *output;
@property (strong, nonatomic) AVCaptureVideoPreviewLayer *layer;

@property (assign, nonatomic) BOOL isHandling;
@property (assign, nonatomic) NSInteger frameNum;
@property (assign, nonatomic) int time;

@end

@implementation VideoController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    NSError *error = nil;
    self.session = [[AVCaptureSession alloc] init];
    self.session.sessionPreset = AVCaptureSessionPresetHigh;
    AVCaptureDevice *frontCamera = nil;
    NSArray *cameras = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *camera in cameras) {
        if (camera.position == AVCaptureDevicePositionFront) {
            frontCamera = camera;
        }
    }
    
    // 用device对象创建一个设备对象input，并将其添加到session
    self.input = [AVCaptureDeviceInput deviceInputWithDevice:frontCamera error:&error];
    [self.session addInput:self.input];

    self.output = [[AVCaptureVideoDataOutput alloc] init];
    self.output.videoSettings = [NSDictionary dictionaryWithObject:[NSNumber numberWithInt:kCVPixelFormatType_32BGRA] forKey:(id)kCVPixelBufferPixelFormatTypeKey];
    [self.session addOutput:self.output];
    
    AVCaptureConnection *connection = [self.output connectionWithMediaType:AVMediaTypeVideo];
    connection.videoOrientation = AVCaptureVideoOrientationPortrait;
            
    dispatch_queue_t queue = dispatch_queue_create("CameraQueue", NULL);
    [_output setSampleBufferDelegate:self queue:queue];
            
    dispatch_async(dispatch_get_main_queue(), ^{
        self.layer = [AVCaptureVideoPreviewLayer layerWithSession:self.session];
        self.layer.videoGravity = AVLayerVideoGravityResizeAspectFill;
        self.layer.frame = CGRectMake(0, 0, self.preview.bounds.size.width, self.preview.bounds.size.height);
        [self.preview.layer addSublayer:self.layer];
        
        [self.session commitConfiguration];
        [self.session startRunning];
    });
}

- (IBAction)close:(id)sender {
    [self.session stopRunning];
    [self dismissViewControllerAnimated:YES completion:^{
    }];
}

- (UIImage *)convertSampleBufferToImage:(CMSampleBufferRef)sampleBuffer {
    // 制作CVImageBufferRef
    CVImageBufferRef buffer;
    buffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    CVPixelBufferLockBaseAddress(buffer, 0);

    // 从 CVImageBufferRef 取得影像的细部信息
    uint8_t *base;
    size_t width, height, bytesPerRow;
    base = CVPixelBufferGetBaseAddress(buffer);
    width = CVPixelBufferGetWidth(buffer);
    height = CVPixelBufferGetHeight(buffer);
    bytesPerRow = CVPixelBufferGetBytesPerRow(buffer);

    // 利用取得影像细部信息格式化 CGContextRef
    CGColorSpaceRef colorSpace;
    CGContextRef cgContext;
    colorSpace = CGColorSpaceCreateDeviceRGB();
    cgContext = CGBitmapContextCreate(base, width, height, 8, bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    CGColorSpaceRelease(colorSpace);

    // 透过 CGImageRef 将 CGContextRef 转换成 UIImage
    CGImageRef cgImage;
    UIImage *image;
    cgImage = CGBitmapContextCreateImage(cgContext);
    image = [UIImage imageWithCGImage:cgImage];
    CGImageRelease(cgImage);
    CGContextRelease(cgContext);

    CVPixelBufferUnlockBaseAddress(buffer, 0);

    return image;
}

- (void)face:(UIImage *)image type:(int)type {
    CGImageRef imageRef = image.CGImage;
    size_t width = CGImageGetWidth(imageRef);
    size_t height = CGImageGetHeight(imageRef);
    NSArray<Box *> *boxes = [self.viewController.mtcnn detectFaces:image minFaceSize:(int)width / 5];
    if (boxes.count == 0) {
        [self setText:-1 score:0];
        self.isHandling = NO;
        return;
    }
    
    Box *box = boxes[0];
    [box toSquareShape];
    if ([box transboundW:(int)width H:(int)height]) {
        [self setText:-1 score:0];
        self.isHandling = NO;
        return;
    }
    
    UIImage *cropImage = [Tools cropImage:image toRect:box.transform2Rect];
    self.time++;
    
    int laplace = [self.viewController.fas laplacian:cropImage];
    if (laplace < laplacian_threshold) {
        [self setText:laplace score:0];
        self.isHandling = NO;
        return;
    }
    
    float score = [self.viewController.fas antiSpoofing:cropImage];
    if (score > fas_threshold) {
        [self setText:laplace score:score];
        self.isHandling = NO;
        return;
    }
    
    float compare = 0;
    if (type == 2) {
        compare = [self.viewController.mfn compare:self.viewController.inputImage with:cropImage];
    }
    
    dispatch_async(dispatch_get_main_queue(), ^{
        if (type == 1) {
            self.viewController.inputImage = cropImage;
            self.viewController.faceView.image = cropImage;
        }
        [self.viewController setText:self.time laplace:laplace score:score compare:compare];
        [self close:nil];
    });
}

- (void)setText:(int)laplace score:(float)score {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (laplace == -1) {
            self.resultLabel.text = @"未检测到人脸";
            return;
        }
        
        NSString *text = [NSString stringWithFormat:@"识别次数：%d\n图片清晰度得分：%d", self.time, laplace];
        if (laplace > laplacian_threshold) {
            text = [text stringByAppendingFormat:@"\n活体检测得分：%f", score];
        }
        self.resultLabel.text = text;
    });
}

#pragma mark - AVCaptureVideoDataOutputSampleBufferDelegate
- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
    
    if (self.frameNum > 5 && !self.isHandling) {
        self.isHandling = YES;
        UIImage *image = [self convertSampleBufferToImage:sampleBuffer];
        [self face:image type:self.type];
    }
    
    self.frameNum++;
}

@end
