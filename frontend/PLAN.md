# Frontend Plan



**Owner:** Siyu Liu (@Siyu-Liu123)



## Goal



Build a user-friendly hotel cancellation prediction interface.



## Tasks



- Build a web interface using Flask and HTML.

- Collect booking information, including hotel type, arrival date, lead time, deposit type, market segment, and ADR. Align all fields with the API input schema.

- Send booking information to the prediction API and display the cancellation probability and risk level.

- Coordinate with the API developer to return the top 3 SHAP factors through the /predict endpoint. The frontend will display these explanations without loading the model.

- Configure the API address through the API_URL environment variable.

- Handle invalid inputs and API failures with user-friendly error messages.

- Add a test to verify error handling when an API call fails.

- Containerize the frontend as a separate Docker service.

- Prepare the final presentation if required.



## Integration



- Proposed Docker service name: frontend.

- Proposed frontend port: 8501.

- Confirm the API URL, input schema, and SHAP output format with teammates.

