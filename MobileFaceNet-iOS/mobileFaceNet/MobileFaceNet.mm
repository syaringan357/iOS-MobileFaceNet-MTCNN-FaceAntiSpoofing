//
//  MobileFaceNet.m
//  MobileFaceNet-iOS
//
//  Created by 周文鹏 on 2019/10/30.
//  Copyright © 2019 周文鹏. All rights reserved.
//

#import "MobileFaceNet.h"
#import "Tools.h"
#import "TFLTensorFlowLite.h"

static NSString * modelFileName = @"MobileFaceNet";
static NSString * modelFileType = @"tflite";

static int image_width = 112; // input图片宽
static int image_height = 112; // input图片高
static int embeddings_size = 192; // 输出的embedding维度


@interface MobileFaceNet()

@property (nonatomic) TFLInterpreter *interpreter;

@end


@implementation MobileFaceNet

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
- (float)compare:(UIImage *)image1 with:(UIImage *)image2 {
    CGSize size = CGSizeMake(image_width, image_height);
    UIImage *imageScale1 = [Tools scaleImage:image1 toSize:size];
    UIImage *imageScale2 = [Tools scaleImage:image2 toSize:size];
    NSData *data = [self dataWithProcessImage1:imageScale1 image2:imageScale2];
  
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
    TFLTensor *outputTensor = [self.interpreter outputTensorAtIndex:0 error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    NSData *outputData = [outputTensor dataWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    
    // 得到前向传播结果
    float *output = new float[2 * embeddings_size];
    [outputData getBytes:output length:(sizeof(float) * 2 * embeddings_size)];
    [self l2Normalize:output epsilon:1e-10];
    return [self evaluate:output];
}

/**
 将两张图片归一化后放入tensor中，由于ios图片是rgba4个通道，所以要过滤掉alpha通道
 */
- (NSData *)dataWithProcessImage1:(UIImage *)image1 image2:(UIImage *)image2 {
    UInt8 *image_data1 = [Tools convertUIImageToBitmapRGBA8:image1];
    UInt8 *image_data2 = [Tools convertUIImageToBitmapRGBA8:image2];
    UInt8 *image_datas[2] = {image_data1, image_data2};
    float *floats = new float[2 * image_width * image_height * 3];
    
    // 将图片归一化后放入tensor中，由于ios图片是rgba4个通道，所以要过滤掉alpha通道
    const float input_mean = 127.5f;
    const float input_std = 128.0f;
    int k = 0;
    for (int i = 0; i < 2; i++) {
        UInt8 *image_data = image_datas[i];
        int size = image_width * image_height * 4;
        for (int j = 0; j < size; j++) {
            if (j % 4 == 3) {
                continue;
            }
            floats[k] = (image_data[j] - input_mean) / input_std;
            k++;
        }
    }
    
    NSData *data = [NSData dataWithBytes:floats length:sizeof(float) * 2 * image_width * image_height * 3];
    return data;
}

/**
 l2范数归一化
 */
- (void)l2Normalize:(float *)embeddings epsilon:(float)epsilon {
    for (int i = 0; i < 2; i++) {
        float square_sum = 0;
        for (int j = 0; j < embeddings_size; j++) {
            square_sum += pow(embeddings[i * embeddings_size + j], 2);
        }
        float x_inv_norm = sqrt(MAX(square_sum, epsilon));
        for (int j = 0; j < embeddings_size; j++) {
            embeddings[i * embeddings_size + j] = embeddings[i * embeddings_size + j] / x_inv_norm;
        }
    }
}

/**
 计算相似度
 */
- (float)evaluate:(float *)embeddings {
    float dist = 0;
    for (int i = 0; i < embeddings_size; i++) {
        dist += pow(embeddings[i] - embeddings[i + embeddings_size], 2);
    }
    float same = 0;
    for (int i = 0; i < 400; i++) {
        float threshold = 0.01f * (i + 1);
        if (dist < threshold) {
            same += 1.0 / 400;
        }
    }
    return same;
}

@end
