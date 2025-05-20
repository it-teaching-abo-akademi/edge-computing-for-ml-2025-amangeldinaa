#include <Arduino_OV767X.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"

tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 140 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const char* GESTURES[] = {
  "rock",
  "paper",
  "scissors"
};
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

unsigned short pixels[176 * 144]; 

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("Initializing camera...");
  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }

  Serial.println("Initializing model...");
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter->AllocateTensors();

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.println("Setup done. Waiting 5 seconds before first capture...");
  delay(5000);
}

void loop() {
  Serial.println("Capturing frame...");
  Camera.readFrame(pixels);

  int numPixels = Camera.width() * Camera.height();
  for (int i = 0; i < numPixels; i++) {
    unsigned short p = pixels[i];
    if (p < 0x1000) {
      Serial.print('0');
    }
    if (p < 0x0100) {
      Serial.print('0');
    }
    if (p < 0x0010) {
      Serial.print('0');
    }
    Serial.print(p, HEX);
  }

  Serial.println("Frame captured. Processing...");

  const int srcW = 176;
  const int srcH = 144;
  const int dstW = 32;
  const int dstH = 32;

  float xScale = (float)srcW / dstW;
  float yScale = (float)srcH / dstH;

  for (int y = 0; y < dstH; y++) {
    for (int x = 0; x < dstW; x++) {
      int srcX = (int)(x * xScale);
      int srcY = (int)(y * yScale);
      unsigned short pixel = pixels[srcY * srcW + srcX];

      uint8_t r = ((pixel >> 11) & 0x1F) << 3;
      uint8_t g = ((pixel >> 5) & 0x3F) << 2;
      uint8_t b = (pixel & 0x1F) << 3;

      uint8_t gray = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);

      int dstIndex = y * dstW + x;
      tflInputTensor->data.uint8[dstIndex] = gray;
    }
  }

  Serial.println("Running inference...");
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(5000);  
    return;
  }

  Serial.println("Results:");
  for (int i = 0; i < NUM_GESTURES; i++) {
    uint8_t value = tflOutputTensor->data.uint8[i];
    float probability = (float)value / 255;
    Serial.print(GESTURES[i]);
    Serial.print(": ");
    Serial.print(value);
    Serial.print(" (probability: ");
    Serial.print(probability, 3);
    Serial.println(")");
  }
  Serial.println();

  delay(10000);  
}