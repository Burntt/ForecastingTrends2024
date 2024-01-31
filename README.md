# Forecasting Trends in Cloud-Edge Computing with Informer

## Overview
This repository is dedicated to the Informer model, a cutting-edge approach designed to forecast cloud-edge computing trends. By leveraging sophisticated attention mechanisms, Informer provides precise and actionable insights into future computational demands and optimization strategies.

## Features
- **Innovative Forecasting**: Employs the Informer model, integrating attention mechanisms for superior accuracy in trend prediction.
- **Tailored for Cloud-Edge**: Specifically designed for the dynamics of cloud and edge computing environments, ensuring relevance and applicability.
- **Efficient and Scalable**: Built to accommodate large datasets and complex computations with high efficiency.

## Installation
1. Confirm Docker is installed and operational on your system.
2. Clone the repository: `git clone [repository-url]`.
3. Navigate to the cloned directory: `cd [repository-name]`.
4. Construct the Docker image: `docker build -t informer .`.
5. Deploy the Docker container: `docker run -d -p 8080:8080 informer`.


## Contributing
Contributions are encouraged! Enhance the project through pull requests, or suggest features and report issues.

## License
This project is under the MIT License. See the LICENSE file for more details.

## Acknowledgments
We extend our gratitude to the authors and contributors of the original Informer GitHub repository for their foundational work, which facilitated the rapid adaptation and implementation of their code into our project. Their innovative approach to long sequence time-series forecasting has been instrumental in guiding our research direction.

## Citations
```
@article{Gort2024TrendsCloudEdgeComputing,
  title={Forecasting Trends in Cloud-Edge Computing: Unleashing the Power of Attention Mechanisms},
  author={Berend J.D. Gort and Maria A. Serrano and Angelos Antonopoulos},
  journal={ArXiv},
  year={2024},
  volume={abs/2401.00001},
  organization={Nearby Computing},
  address={Barcelona, Spain},
  email={{berend.gort, maria.serrano, angelos.antonopoulos}@nearbycomputing.com},
  abstract={
    "This project introduces an innovative approach to forecast trends in cloud-edge computing using advanced attention mechanisms. Our model, named 'Informer', leverages deep learning to analyze temporal data, providing insights into future computing needs and optimizations."
  },
  keywords={Zero-touch, AI, LSTM, Transformers, Informers, Edge Computing, Time-series Forecasting, Deep Learning}
}
```