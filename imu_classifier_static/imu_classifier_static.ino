#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
 
#include "FIQ_PrunedConV_model.h"
#include "output_scissors.h"

tflite::AllOpsResolver tflOpsResolver;
 
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;
 
constexpr int tensorArenaSize = 140 * 1024;
attribute((aligned(16))) byte tensorArena[tensorArenaSize];
 
const char* GESTURES[] = {
  "rock",
  "paper",
  "scissors"
};
 
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))
 
void setup() {
  Serial.begin(9600);
  while (!Serial);
 
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }
 
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
 
  tflInterpreter->AllocateTensors();

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}
 
void loop() {
 
  int i=0;

  for(int i = 0; i < (32 * 32); i++)
  {
    tflInputTensor->data.uint8[i] = image_data[i];
  }

  unsigned long startMicros = micros();
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  unsigned long endMicros = micros();
  unsigned long inferenceTime = endMicros - startMicros;
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed!");
    while (1);
    return;
  }

  Serial.print("Inference time: ");
  Serial.print(inferenceTime);
  Serial.println(" microseconds");

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

  delay(20000);
}