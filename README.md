
# 🚛  Average Fuel Consumption of Heavy Vehicles Using Machine Learning

This project implements a **Machine Learning–based prediction system** that estimates **average fuel consumption of heavy vehicles** using sensor-based parameters. The system is built using **Python Tkinter GUI** and leverages **ANN, SVM, and Random Forest** algorithms for model training and comparison.

---

## 🧾 Published Research – Official Recognition

This project is **officially published** in the _International Scientific Journal of Engineering & Management (ISJEM)._

📜 **Publication Certificate Details:**  
- **Author:** E. Pavan Kumar  
- **Title:** _Average Fuel Consumption of Heavy Vehicles Using Machine Learning_  
- **Journal:** International Scientific Journal of Engineering & Management (ISJEM)  
- **Volume / Issue:** Vol. 04, Issue 05  
- **Publication Date:** May 2025  
- **ISSN:** 2583–6129  
- **Impact Factor:** 7.839  
- **DOI:** 10.55041/ISJEM03813  

> 🏆 This certificate verifies successful publication and recognition at journal level.

---

## 🧠 Project Overview

Fuel consumption is a critical cost and efficiency factor in heavy-vehicle fleet operations. This system predicts average fuel usage based on seven predictors aggregated per vehicle distance window.

The Python application provides:
- GUI to upload dataset and run model
- ANN model using TensorFlow
- SVM & Random Forest comparison
- Fuel prediction for new uploaded data
- Visual graph outputs

---

## 🗂 Dataset Description

Dataset Sample Fields:
```
num_stops, time_stopped, avg_moving_speed, characteristic_acceleration,
aerodynamic_speed_squared, change_in_kinetic_energy, change_in_potential_energy, class
```

| Feature | Description |
|--------|-------------|
| num_stops | Number of stops in travel window |
| time_stopped | Time vehicle remains idle |
| avg_moving_speed | Average vehicle speed |
| characteristic_acceleration | Acceleration dynamics |
| aerodynamic_speed_squared | Aerodynamic coefficient |
| change_in_kinetic_energy | Power variation |
| change_in_potential_energy | Height / terrain factor |
| class | Target output – predicted fuel level |

---

## 🧪 Algorithms Used

| Algorithm | Purpose |
|----------|---------|
| ANN (TensorFlow) | Primary model for prediction |
| SVM (RBF kernel) | Comparison classifier |
| Random Forest | Comparison model |

---

## 🛠 Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
python Main.py
```

---

## ▶ Running The App
- Upload dataset
- Train model
- Compare ANN, RF, SVM
- Perform predictions
- View consumption graphs

---

## 📈 Output Example
```
ANN Accuracy: 91.2%
SVM Accuracy: 88.4%
RF Accuracy: 83.7%
```

---

## 🔧 Future Enhancements
- Streamlit or Django Web Deployment
- Real-time IoT fleet input model
- Fuel performance report export (PDF/Excel)

---

## 🤝 Contributors
| Name | Role |
|------|------|
| E. Pavan Kumar | Developer / Research Author |

---

## 📜 License
MIT License
