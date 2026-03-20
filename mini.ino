const int PIN_X = 9;   // Horizontal: LEFT / RIGHT
const int PIN_Y = 10;  // Vertical:   UP / DOWN

void setup() {
  pinMode(PIN_X, OUTPUT);
  pinMode(PIN_Y, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() >= 2) {
    char cmdX = Serial.read();
    char cmdY = Serial.read();

    // --- X axis (pin 9) ---
    if (cmdX == 'R') {
      pinMode(PIN_X, OUTPUT);
      analogWrite(PIN_X, 255);
      Serial.println(">> RIGHT");
    } else if (cmdX == 'L') {
      pinMode(PIN_X, OUTPUT);
      analogWrite(PIN_X, 0);
      Serial.println("<< LEFT");
    } else if (cmdX == 'S') {
      pinMode(PIN_X, INPUT);
      Serial.println("-- X OFF");
    }

    // --- Y axis (pin 10) ---
    if (cmdY == 'D') {
      pinMode(PIN_Y, OUTPUT);
      analogWrite(PIN_Y, 255);
      Serial.println("vv DOWN");
    } else if (cmdY == 'U') {
      pinMode(PIN_Y, OUTPUT);
      analogWrite(PIN_Y, 0);
      Serial.println("^^ UP");
    } else if (cmdY == 'S') {
      pinMode(PIN_Y, INPUT);
      Serial.println("-- Y OFF");
    }
  }
}