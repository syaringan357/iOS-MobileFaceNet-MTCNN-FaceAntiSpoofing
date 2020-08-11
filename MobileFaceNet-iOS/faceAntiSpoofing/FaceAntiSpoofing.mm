//
//  FaceAntiSpoofing.m
//  MobileFaceNet-iOS
//
//  Created by 张建伟 on 2020/1/22.
//  Copyright © 2020 周文鹏. All rights reserved.
//

#import "FaceAntiSpoofing.h"
#import "Tools.h"
#import "TFLTensorFlowLite.h"

static NSString * modelFileName = @"FaceAntiSpoofing";
static NSString * modelFileType = @"tflite";

static int image_width = 256; // input图片宽
static int image_height = 256; // input图片高


@interface FaceAntiSpoofing()

@property (nonatomic) TFLInterpreter *interpreter;

@end


@implementation FaceAntiSpoofing

/**
 初始化
 */
- (instancetype)init {
    if (self = [super init]) {
        NSString *modelPath = [Tools filePathForResourceName:modelFileName extension:modelFileType];
        TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
        options.numberOfThreads = 4;
        NSError *error;
        self.interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath options:options error:&error];
        if (error) {
            NSLog(@"%@", error);
        }
        [self.interpreter allocateTensorsWithError:&error];
        if (error) {
            NSLog(@"%@", error);
        }
    }
    return self;
}

/**
 比较两张人脸图片
 */
- (float)antiSpoofing:(UIImage *)image {
    CGSize size = CGSizeMake(image_width, image_height);
    UIImage *imageScale = [Tools scaleImage:image toSize:size];
    NSData *data = [self dataWithProcessImage:imageScale];
  
    // 前向传播
    NSError *error;
    TFLTensor *inputTensor = [self.interpreter inputTensorAtIndex:0 error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [inputTensor copyData:data error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [self.interpreter invokeWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    TFLTensor *clss_pred_tensor = [self.interpreter outputTensorAtIndex:0 error:&error];
    TFLTensor *leaf_node_mask_tensor = [self.interpreter outputTensorAtIndex:1 error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    NSData *clss_pred_data = [clss_pred_tensor dataWithError:&error];
    NSData *leaf_node_mask_data = [leaf_node_mask_tensor dataWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    
    // 得到前向传播结果
    float *clss_pred = new float[8];
    float *leaf_node_mask = new float[8];
    [clss_pred_data getBytes:clss_pred length:(sizeof(float) * 8)];
    [leaf_node_mask_data getBytes:leaf_node_mask length:(sizeof(float) * 8)];
    
    float score = 0;
    for (int i = 0; i < 8; i++) {
        score += abs(clss_pred[i]) * leaf_node_mask[i];
    }
    delete [] clss_pred;
    delete [] leaf_node_mask;
    
    return score;
}

/**
 将图片归一化后放入tensor中，由于ios图片是rgba4个通道，所以要过滤掉alpha通道
 */
- (NSData *)dataWithProcessImage:(UIImage *)image {
    UInt8 *image_data = [Tools convertUIImageToBitmapRGBA8:image];
    float *floats = new float[image_width * image_height * 3];
    
    // 将图片归一化后放入tensor中，由于ios图片是rgba4个通道，所以要过滤掉alpha通道
    const float input_std = 255.0f;
    int k = 0;
    int size = image_width * image_height * 4;
    for (int j = 0; j < size; j++) {
        if (j % 4 == 3) {
            continue;
        }
        floats[k] = image_data[j] / input_std;
        k++;
    }
    free(image_data);
    
    NSData *data = [NSData dataWithBytes:floats length:sizeof(float) * image_width * image_height * 3];
    delete [] floats;
    
    return data;
}

@end
