#include <Servo.h>
int servoPin = 4; // hip
int servoPink = 6; // knee
Servo Servo1;
Servo Servo2;
void setup() {
  Servo1.attach(servoPin);
  Servo1.write(30);
  delay(1000);
  Servo2.attach(servoPink);
  Servo2.write =(150);
  delay(1000);
}
int HFF = 30;
int HFE = 150;
float hip_rate = 0.8; //0.1
float hip_angle = HFF; //0
int HF = 0;
int HE = 1;
int KFF = 30;
int KFE = 150;
float knee_rate = 0.8;
float knee_angle = KFF;
int KF = 0;
int KE = 1;
void loop() {
  if (hip_angle > HFF && HF) {
    hip_angle -=1;
    Servo1.write(hip_angle);
    delay((1.2/(hip_rate*120.0))*500);
    if (hip_angle <= HFF) {
      HF = 0;
      HE = 1;
    }
  }
  else if (hip_angle < HFE && HE) {
    hip_angle +=1;
    Servo1.write(hip_angle);
    delay((1.2/(hip_rate*120.0))*500);
    if (hip_angle >= HFE) {
      HF = 1;
      HE = 0;
    }
   }
  if (knee_angle > KFF && KF) {
    knee_angle -=1;
    Servo1.write(knee_angle);
    delay((1.2/(knee_rate*120.0))*500);
    if (knee_angle <= KFF) {
      KF = 0;
      KE = 1;
    }
  else if (knee_angle < KFE && KE) {
    knee_angle +=1;
    Servo1.write(knee_angle);
    delay((1.2/(hip_rate*120.0))*500);
    if (knee_angle >= KFE) {
      KF = 1;
      KE = 0;
    }