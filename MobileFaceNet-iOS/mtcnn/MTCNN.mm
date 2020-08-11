//
//  MTCNN.m
//  MobileFaceNet-iOS
//
//  Created by 张建伟 on 2020/1/16.
//  Copyright © 2020 周文鹏. All rights reserved.
//

#import "MTCNN.h"
#import "Tools.h"
#import "TFLTensorFlowLite.h"

static NSString *pnetModelFileName = @"pnet";
static NSString *rnetModelFileName = @"rnet";
static NSString *onetModelFileName = @"onet";
static NSString *modelFileType = @"tflite";

static float factor = 0.709f;
static float p_net_threshold = 0.6f;
static float r_net_threshold = 0.7f;
static float o_net_threshold = 0.7f;


@interface MTCNN()

@property (nonatomic) TFLInterpreter *pnetInterpreter;
@property (nonatomic) TFLInterpreter *rnetInterpreter;
@property (nonatomic) TFLInterpreter *onetInterpreter;

@end


@implementation MTCNN

/**
 初始化
 */
- (instancetype)init {
    if (self = [super init]) {
        TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
        options.numberOfThreads = 4;
        NSError *error;
        
        NSString *pnetModelPath = [Tools filePathForResourceName:pnetModelFileName extension:modelFileType];
        self.pnetInterpreter = [[TFLInterpreter alloc] initWithModelPath:pnetModelPath options:options error:&error];
        if (error) {
            NSLog(@"%@", error);
        }
        
        NSString *rnetModelPath = [Tools filePathForResourceName:rnetModelFileName extension:modelFileType];
        self.rnetInterpreter = [[TFLInterpreter alloc] initWithModelPath:rnetModelPath options:options error:&error];
        if (error) {
            NSLog(@"%@", error);
        }
        
        NSString *onetModelPath = [Tools filePathForResourceName:onetModelFileName extension:modelFileType];
        self.onetInterpreter = [[TFLInterpreter alloc] initWithModelPath:onetModelPath options:options error:&error];
        if (error) {
            NSLog(@"%@", error);
        }
    }
    return self;
}

/**
 检测人脸
 */
- (NSArray<Box *> *)detectFaces:(UIImage *)image minFaceSize:(int)minFaceSize {
    int width = (int) CGImageGetWidth(image.CGImage);
    int height = (int) CGImageGetHeight(image.CGImage);
    
    //【1】PNet generate candidate boxes
    NSMutableArray<Box *> *boxes = [self pNet:image minSize:minFaceSize];
    [self squareLimit:boxes W:width H:height];
    
    //【2】RNet
    boxes = [self rNet:image boxes:boxes];
    [self squareLimit:boxes W:width H:height];
    
    //【3】ONet
    boxes = [self oNet:image boxes:boxes];
    return boxes;
}

/**
 pnet预处理和结果处理
 */
- (NSMutableArray<Box *> *)pNet:(UIImage *)image minSize:(int)minSize {
    int whMin = MIN((int) CGImageGetWidth(image.CGImage), (int) CGImageGetHeight(image.CGImage));
    float currentFaceSize = minSize; // currentFaceSize=minSize/(factor^k) k=0,1,2... until excced whMin
    NSMutableArray<Box *> *totalBoxes = [NSMutableArray array];
    
    //【1】Image Paramid and Feed to Pnet
    while (currentFaceSize <= whMin) {
        float scale = 12.0f / currentFaceSize;
        
        // (1)Image Scale
        UIImage *img = [Tools scaleImage:image toScale:scale];
        int w = (int) CGImageGetWidth(img.CGImage);
        int h = (int) CGImageGetHeight(img.CGImage);
        
        // (2)RUN CNN
        int outW = ceil(w * 0.5 - 5) + 0.5;
        int outH = ceil(h * 0.5 - 5) + 0.5;
        float *prob1 = new float[outW * outH * 2];
        float *conv4_2_BiasAdd = new float[outW * outH * 4];
        [self pNetForward:img W:w H:h Prob:prob1 Bias:conv4_2_BiasAdd OutW:outW OutH:outH];
        
        // (3)数据解析
        NSMutableArray<Box *> *curBoxes = [NSMutableArray array];
        [self generateBoxes:curBoxes Prob:prob1 Bias:conv4_2_BiasAdd Scale:scale W:outW H:outH];
        
        // (4)nms 0.5
        [self nms:curBoxes Threshold:0.5 Method:@"Union"];
        
        // (5)add to totalBoxes
        for (int i = 0; i < curBoxes.count; i++) {
            if (!curBoxes[i].deleted) {
                [totalBoxes addObject:curBoxes[i]];
            }
        }
        
        // Face Size等比递增
        currentFaceSize /= factor;
        
        delete [] prob1;
        delete [] conv4_2_BiasAdd;
    }
    
    // NMS 0.7
    [self nms:totalBoxes Threshold:0.7f Method:@"Union"];
    
    // BBR
    [self boundingBoxReggression:totalBoxes];
    
    return [self updateBoxes:totalBoxes];
}

/**
 pnet前向传播
 */
- (void)pNetForward:(UIImage *)image W:(int)w H:(int)h Prob:(float *)prob1 Bias:(float *)conv4_2_BiasAdd OutW:(int)outW OutH:(int)outH {
    
    float *floats = [self normalizeImage:image W:w H:h];
    [self transpose:floats H:h W:w C:3];  // 转置
    NSData *data = [NSData dataWithBytes:floats length:sizeof(float) * w * h * 3];
    delete [] floats;

    // 前向传播
    NSArray<NSNumber *> *shape = @[@1, @(w), @(h), @3];
    NSError *error;
    [self.pnetInterpreter resizeInputTensorAtIndex:0 toShape:shape error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [self.pnetInterpreter allocateTensorsWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    TFLTensor *inputTensor = [self.pnetInterpreter inputTensorAtIndex:0 error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [inputTensor copyData:data error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [self.pnetInterpreter invokeWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    TFLTensor *prob1_tensor = [self.pnetInterpreter outputTensorAtIndex:0 error:&error];
    TFLTensor *conv4_2_BiasAdd_tensor = [self.pnetInterpreter outputTensorAtIndex:1 error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    NSData *prob1_data = [prob1_tensor dataWithError:&error];
    NSData *conv4_2_BiasAdd_data = [conv4_2_BiasAdd_tensor dataWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    
    // 得到前向传播输出
    [prob1_data getBytes:prob1 length:(sizeof(float) * outW * outH * 2)];
    [conv4_2_BiasAdd_data getBytes:conv4_2_BiasAdd length:(sizeof(float) * outW * outH * 4)];
    
    // 转置
    [self transpose:prob1 H:outW W:outH C:2];
    [self transpose:conv4_2_BiasAdd H:outW W:outH C:4];
}

/**
 rnet预处理和结果处理
 */
- (NSMutableArray<Box *> *)rNet:(UIImage *)image boxes:(NSMutableArray<Box *> *)boxes {
    // RNet Input Init
    int num = (int) boxes.count;
    float *floats = new float[num * 24 * 24 * 3];
    int k = 0;
    for (int i = 0; i < num; i++) {
        UIImage *img = [self crop:image withBox:boxes[i] andScale:24];
        float *img_data = [self normalizeImage:img W:24 H:24];
        [self transpose:img_data H:24 W:24 C:3];  // 转置
        for (int j = 0; j < 24 * 24 * 3; j++) {
            floats[k] = img_data[j];
            k++;
        }
        delete [] img_data;
    }
    NSData *data = [NSData dataWithBytes:floats length:sizeof(float) * num * 24 * 24 * 3];
    delete [] floats;
    
    // Run RNet
    [self rNetForward:data boxes:boxes];
    
    // RNetThreshold
    for (int i = 0; i < num; i++) {
        if (boxes[i].score < r_net_threshold) {
            boxes[i].deleted = true;
        }
    }
    
    // Nms 0.7
    [self nms:boxes Threshold:0.7f Method:@"Union"];
    [self boundingBoxReggression:boxes];
    return [self updateBoxes:boxes];
}

/**
 rnet前向传播
 */
- (void)rNetForward:(NSData *)data boxes:(NSMutableArray<Box *> *)boxes {
    int num = (int) boxes.count;
    
    // 前向传播
    NSError *error;
    NSArray<NSNumber *> *shape = @[@(num), @24, @24, @3];
    [self.rnetInterpreter resizeInputTensorAtIndex:0 toShape:shape error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [self.rnetInterpreter allocateTensorsWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    TFLTensor *inputTensor = [self.rnetInterpreter inputTensorAtIndex:0 error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [inputTensor copyData:data error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [self.rnetInterpreter invokeWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    TFLTensor *prob1_tensor = [self.rnetInterpreter outputTensorAtIndex:0 error:&error];
    TFLTensor *conv5_2_conv5_2_tensor = [self.rnetInterpreter outputTensorAtIndex:1 error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    NSData *prob1_data = [prob1_tensor dataWithError:&error];
    NSData *conv5_2_conv5_2_data = [conv5_2_conv5_2_tensor dataWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    
    // 得到前向传播输出
    float *prob1 = new float[num * 2];
    float *conv5_2_conv5_2 = new float[num * 4];
    [prob1_data getBytes:prob1 length:(sizeof(float) * num * 2)];
    [conv5_2_conv5_2_data getBytes:conv5_2_conv5_2 length:(sizeof(float) * num * 4)];
    
    // 转换
    for (int i = 0; i < num; i++) {
        boxes[i].score = prob1[i * 2 + 1];
        for (int j = 0; j < 4; j++) {
            boxes[i].bbr[j] = @(conv5_2_conv5_2[i * 4 + j]);
        }
    }
    delete [] prob1;
    delete [] conv5_2_conv5_2;
}

/**
 onet预处理和结果处理
 */
- (NSMutableArray<Box *> *)oNet:(UIImage *)image boxes:(NSMutableArray<Box *> *)boxes {
    // ONet Input Init
    int num = (int) boxes.count;
    float *floats = new float[num * 48 * 48 * 3];
    int k = 0;
    for (int i = 0; i < num; i++) {
        UIImage *img = [self crop:image withBox:boxes[i] andScale:48];
        float *img_data = [self normalizeImage:img W:48 H:48];
        [self transpose:img_data H:48 W:48 C:3];  // 转置
        for (int j = 0; j < 48 * 48 * 3; j++) {
            floats[k] = img_data[j];
            k++;
        }
        delete [] img_data;
    }
    NSData *data = [NSData dataWithBytes:floats length:sizeof(float) * num * 48 * 48 * 3];
    delete [] floats;
    
    // Run ONet
    [self oNetForward:data boxes:boxes];
    
    // ONetThreshold
    for (int i = 0; i < num; i++) {
        if (boxes[i].score < o_net_threshold) {
            boxes[i].deleted = true;
        }
    }
    
    [self boundingBoxReggression:boxes];
    // Nms 0.7
    [self nms:boxes Threshold:0.7f Method:@"Min"];
    return [self updateBoxes:boxes];
}

/**
 onet前向传播
 */
- (void)oNetForward:(NSData *)data boxes:(NSMutableArray<Box *> *)boxes {
    int num = (int) boxes.count;
    
    // 前向传播
    NSError *error;
    NSArray<NSNumber *> *shape = @[@(num), @48, @48, @3];
    [self.onetInterpreter resizeInputTensorAtIndex:0 toShape:shape error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [self.onetInterpreter allocateTensorsWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    TFLTensor *inputTensor = [self.onetInterpreter inputTensorAtIndex:0 error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [inputTensor copyData:data error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    [self.onetInterpreter invokeWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    TFLTensor *prob1_tensor = [self.onetInterpreter outputTensorAtIndex:0 error:&error];
    TFLTensor *conv6_2_conv6_2_tensor = [self.onetInterpreter outputTensorAtIndex:1 error:&error];
    TFLTensor *conv6_3_conv6_3_tensor = [self.onetInterpreter outputTensorAtIndex:2 error:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    NSData *prob1_data = [prob1_tensor dataWithError:&error];
    NSData *conv6_2_conv6_2_data = [conv6_2_conv6_2_tensor dataWithError:&error];
    NSData *conv6_3_conv6_3_data = [conv6_3_conv6_3_tensor dataWithError:&error];
    if (error) {
        NSLog(@"%@", error);
    }
    
    // 得到前向传播输出
    float *prob1 = new float[num * 2];
    float *conv6_2_conv6_2 = new float[num * 4];
    float *conv6_3_conv6_3 = new float[num * 10];
    [prob1_data getBytes:prob1 length:(sizeof(float) * num * 2)];
    [conv6_2_conv6_2_data getBytes:conv6_2_conv6_2 length:(sizeof(float) * num * 4)];
    [conv6_3_conv6_3_data getBytes:conv6_3_conv6_3 length:(sizeof(float) * num * 10)];
    
    // 转换
    for (int i = 0; i < num; i++) {
        // prob
        boxes[i].score = prob1[i * 2 + 1];
        
        // bias
        for (int j = 0; j < 4; j++) {
            boxes[i].bbr[j] = @(conv6_2_conv6_2[i * 4 + j]);
        }
        
        // landmark
        for (int j = 0; j < 5; j++) {
            int x = round(boxes[i].left + (conv6_3_conv6_3[i * 10 + j] * boxes[i].width));
            int y = round(boxes[i].right + (conv6_3_conv6_3[i * 10 + j + 5] * boxes[i].height));
            boxes[i].landmark[j] = @(CGPointMake(x, y));
        }
    }
    delete [] prob1;
    delete [] conv6_2_conv6_2;
    delete [] conv6_3_conv6_3;
}

/**
 生成boxes
 */
- (void)generateBoxes:(NSMutableArray<Box *> *)boxes Prob:(float *)prob1 Bias:(float *)conv4_2_BiasAdd Scale:(float)scale W:(int)w H:(int)h {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float score = prob1[(y * w + x) * 2 + 1];
            // only accept prob >threadshold(0.6 here)
            if (score > p_net_threshold) {
                Box *box = [[Box alloc] init];
                // core
                box.score = score;
                // box
                box.box[0] = @(round(x * 2 / scale));
                box.box[1] = @(round(y * 2 / scale));
                box.box[2] = @(round((x * 2 + 11) / scale));
                box.box[3] = @(round((y * 2 + 11) / scale));
                // bbr
                for (int i = 0; i < 4; i++) {
                    box.bbr[i] = @(conv4_2_BiasAdd[(y * w + x) * 2 + i]);
                }
                // add
                [boxes addObject:box];
            }
        }
    }
}

/**
 nms
 */
- (void)nms:(NSMutableArray<Box *> *)boxes Threshold:(float)threshold Method:(NSString *)method {
    for (int i = 0; i < boxes.count; i++) {
        Box *box = boxes[i];
        if (!box.deleted) {
            for (int j = i + 1; j < boxes.count; j++) {
                Box *box2 = boxes[j];
                if (!box2.deleted) {
                    int x1 = MAX([box.box[0] intValue], [box2.box[0] intValue]);
                    int y1 = MAX([box.box[1] intValue], [box2.box[1] intValue]);
                    int x2 = MIN([box.box[2] intValue], [box2.box[2] intValue]);
                    int y2 = MIN([box.box[3] intValue], [box2.box[3] intValue]);
                    if (x2 < x1 || y2 < y1) {
                        continue;
                    }
                    int areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1);
                    float iou = 0.0;
                    if ([method isEqualToString:@"Union"]) {
                        iou = 1.0f * areaIoU / ([box area] + [box2 area] - areaIoU);
                    } else if ([method isEqualToString:@"Min"]) {
                        iou = 1.0f * areaIoU / (MIN([box area], [box2 area]));
                    }
                    if (iou >= threshold) {
                        if (box.score > box2.score) {
                            box2.deleted = true;
                        } else {
                            box.deleted = true;
                        }
                    }
                }
            }
        }
    }
}

- (void)boundingBoxReggression:(NSMutableArray<Box *> *)boxes {
    for (int i = 0; i < boxes.count; i++) {
        [boxes[i] calibrate];
    }
}

- (NSMutableArray<Box *> *)updateBoxes:(NSMutableArray<Box *> *)boxes {
    NSMutableArray<Box *> *newBoxes = [NSMutableArray array];
    for (int i = 0; i < boxes.count; i++) {
        if (!boxes[i].deleted) {
            [newBoxes addObject:boxes[i]];
        }
    }
    return newBoxes;
}

- (void)squareLimit:(NSMutableArray<Box *> *)boxes W:(int)w H:(int)h {
    for (int i = 0; i < boxes.count; i++) {
        [boxes[i] toSquareShape];
        [boxes[i] limitSquareW:w H:h];
    }
}

/**
 裁剪并缩放图片
 */
- (UIImage *)crop:(UIImage *)image withBox:(Box *)box andScale:(int)size {
    UIImage *cropped = [Tools cropImage:image toRect:[box transform2Rect]];
    UIImage *scaled = [Tools scaleImage:cropped toSize:CGSizeMake(size, size)];
    return scaled;
}

/**
 将图片归一化后放入tensor中，由于ios图片是rgba4个通道，所以要过滤掉alpha通道
 */
- (float *)normalizeImage:(UIImage *)image W:(int)w H:(int)h {
    UInt8 *image_data = [Tools convertUIImageToBitmapRGBA8:image];
    float *floats = new float[w * h * 3];
    
    // 将图片归一化后放入tensor中，由于ios图片是rgba4个通道，所以要过滤掉alpha通道
    const float input_mean = 127.5f;
    const float input_std = 128.0f;
    int k = 0;
    int size = w * h * 4;
    for (int j = 0; j < size; j++) {
        if (j % 4 == 3) {
            continue;
        }
        floats[k] = (image_data[j] - input_mean) / input_std;
        k++;
    }
    free(image_data);
    
    return floats;
}

/**
 转置
 */
- (void)transpose:(float *)data H:(int)h W:(int)w C:(int)c {
    float *tmp = new float[w * h * c];
    for (int i = 0; i < w * h * c; i++) {
        tmp[i] = data[i];
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int z = 0; z < c; z++) {
                data[(x * h + y) * c + z] = tmp[(y * w + x) * c + z];
            }
        }
    }
    delete [] tmp;
}

@end
