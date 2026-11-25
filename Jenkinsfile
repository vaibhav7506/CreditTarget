pipeline {
    agent any

    tools {
        jdk 'OpenJDK-17'
    }

    environment {
        PYSPARK_PYTHON = 'python3'
        PYSPARK_DRIVER_PYTHON = 'python3'
        STORAGE_BACKEND = 'local'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                sh '''
                    python3.11 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -e ".[dev]"
                '''
            }
        }

        stage('Lint & Type Check') {
            steps {
                sh '''
                    . .venv/bin/activate
                    ruff check src tests
                    mypy src
                '''
            }
        }

        stage('PySpark & Chispa Unit Tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pytest tests -v --cov=credittarget --cov-report=xml
                '''
            }
        }

        stage('E2E Lakehouse & Governance Gates') {
            steps {
                sh '''
                    . .venv/bin/activate
                    credittarget run-all --config configs/local.yaml
                '''
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'artifacts/**/*', fingerprint: true
            }
        }
    }

    post {
        always {
            cleanWs deleteDirs: true, notFailBuild: true
        }
    }
}
