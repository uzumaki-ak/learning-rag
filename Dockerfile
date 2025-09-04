#base image
From python:3.10-slim


#set the environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GEMINI_API_KEY="your_api_key_here"

# SET WORING DIRECTORY
WORKDIR /app

#copy  requirements file
COPY ./app

#install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#exposing streamlit default port
EXPOSE 8501

#Run streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0"]