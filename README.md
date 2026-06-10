\# 🏠 End-to-End Property Valuation Platform



A full-stack, production-ready Machine Learning application that predicts property real estate prices based on key property characteristics. This project features a trained predictive model, an interactive user interface dashboard, complete environment isolation via Docker containerization, and a fully public cloud deployment.



\## 🚀 Live Demo

The application is deployed and running live in the cloud. You can interact with the predictive dashboard here:

👉 \*\*\[Live App Link](https://huggingface.co/spaces/MHR-12/property-valuation-app)\*\*



\---



\## 🛠️ Tech Stack \& Skills Demonstrated

\* \*\*Machine Learning:\*\* Scikit-Learn, Random Forest Regressor, Pandas, NumPy.

\* \*\*Frontend Interface:\*\* Streamlit (Python-native web framework).

\* \*\*DevOps \& Containerization:\*\* Docker (Multi-stage layer builds, environment isolation).

\* \*\*Cloud Infrastructure:\*\* Hugging Face Spaces (Docker SDK runtime setup).



\---



\## 📐 Architecture \& System Design

This project was built from \*\*First Principles\*\* to ensure that environment configuration mismatches between local development and cloud servers are completely eliminated. 



Instead of relying on a raw local Python interpreter, the entire application runtime workspace is bundled into an isolated, reproducible Linux container.



+---------------------------------------+

&#x20;             |         Hugging Face Cloud Server     |

&#x20;             |  +---------------------------------+  |

User Interface --->  |       Streamlit Web App         |  |

(Web Browser)     |  |          (api/app.py)           |  |

|  +---------------+-----------------+  |

|                  | Loads Weights      |

|                  v                    |

|  +---------------------------------+  |

|  |      Random Forest Model        |  |

|  |  (models/random\_forest.pkl)     |  |

|  +---------------------------------+  |

||
|-|

. \*\*The Core Brain:\*\* A robust Machine Learning model trained using a Random Forest architecture.

2\. \*\*The Interface:\*\* A Streamlit UI that collects dynamic inputs (e.g., room counts, square footage, geographic features) from the user.

3\. \*\*The Deployment Layer:\*\* A unified `Dockerfile` that packages the application layers and exposes port `7860` for production hosting.



\---



\## 📂 Project Directory Structure

```text

├── api/

│   └── app.py            # Streamlit dashboard UI logic

├── models/

│   └── random\_forest.pkl # Pre-trained ML model weights file

├── Dockerfile            # Unified Docker blueprint for production deployment

├── requirements.txt      # Comprehensive Python environment dependencies

├── Dockerfile.api        # Backend API blueprint (retained for microservice testing)

├── Dockerfile.app        # Frontend app blueprint (retained for local scaling)

└── docker-compose.yml    # Local multi-container network configuration script



Running Locally with Docker

If you have Docker desktop installed on your computer, you can run this entire production setup locally in seconds without manual package installation.

Clone the Repository:

git clone \[https://github.com/Hassan12557/Property-Price-Detection.git](https://github.com/Hassan12557/Property-Price-Detection.git)

cd Property-Price-Detection



Build the Container:

docker build -t property-valuation-app .

Run the App:

docker run -p 7860:7860 property-valuation-app





Cloud Deployment Configuration

This repository is optimized for cloud deployment via Hugging Face Spaces using the Docker SDK.



Metadata Configuration

The container uses a custom YAML header at the top of the repository README.md to trigger the automatic cloud build sequence:

\---

title: Property Valuation App

emoji: 🏠

colorFrom: blue

colorTo: green

sdk: docker

pinned: false

\---



Production Dockerfile

The deployment runs on a minimized python:3.11-slim Linux image utilizing strict build optimization flags:



FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \&\& apt-get install -y build-essential \&\& rm -rf /var/lib/apt/lists/\*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD \["streamlit", "run", "api/app.py", "--server.port=7860", "--server.address=0.0.0.0"]

