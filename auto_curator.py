import pickle
from datetime import datetime
import smtplib
import numpy as np
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Declare constants
SYSTEM_EMAIL = "globe@example.com"
SYSTEM_EMAIL_PASSWORD = ""
GLOBE_ADMIN_EMAIL = "globe_admin@example.com"
SMTP_SERVER = "smtp.example.com"
SMTP_PORT = 587
TRUSTED_SITES = []
RF_MIN_ACCURACY = 0.7
RF_FEATURES = []


def send_email(recipient_email, subject, body):
    # Prepare email
    message = MIMEMultipart()
    message["From"] = SYSTEM_EMAIL
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Send email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp_server:
            smtp_server.starttls()
            smtp_server.login(SYSTEM_EMAIL, SYSTEM_EMAIL_PASSWORD)
            smtp_server.send_message(message)
        print(f"Notification email sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")


def prepare_rf_data(df, measurement_device):

    # Create feature dataframe
    feature_df = df[RF_FEATURES + ['weight']] if 'weight' in df.columns else df[RF_FEATURES]

    # Create target series
    if measurement_device == 'disk':
        target = df['transparencies:transparency disk image disappearance (m)']
    elif measurement_device == 'tube':
        target = df['transparencies:tube image disappearance (cm)']
    else:
        raise ValueError(f"Invalid measurement device, {measurement_device}")

    return feature_df, target


def random_forest_model(df):
    return df


def bayesian_hierarchical_model(df):
    return df


def random_forest_model_test(df, measurement_device, apply_weight=False):

    # Create feature dataframe and target series
    feature_df, target = prepare_rf_data(df, measurement_device)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(feature_df, target, test_size=0.2, random_state=42)

    # Initialize and fit model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    if apply_weight:
        weights = X_train['weight'].copy()
        X_train.drop('weight', axis=1, inplace=True)
        rf_model.fit(X_train, y_train, weights)
        # Print R2 score
        X_test.drop('weight', axis=1, inplace=True)
    else:
        rf_model.fit(X_train, y_train)

    return rf_model


# Import datasheet
main_df = pd.read_csv('')

# Create dataframe for holding inaccurate samples
columns = list(main_df.columns) + ['predictions:transparency disk image disappearance (m)',
                                   'predictions:tube image disappearance (cm)', 'sample_status']
pending_review_df = pd.DataFrame(columns=columns)

# Add prediction columns if they do not exist
if 'predictions:transparency disk image disappearance (m)' not in main_df.columns:
    main_df['predictions:transparency disk image disappearance (m)'] = np.nan
if 'predictions:tube image disappearance (cm)' not in main_df.columns:
    main_df['predictions:tube image disappearance (cm)'] = np.nan

# Add unique sample ID column if it does not exist
if 'usid' not in main_df.columns:
    main_df['usid'] = np.NAN

# Add sample status column if it does not exist
if 'sample_status' not in main_df.columns:
    main_df['sample_status'] = 'active'


# 0. Prepare model and new sample for testing

# Create new sample with tube measurement
current_time = datetime.now()
new_sample = main_df.loc[25555].to_dict()
new_sample['transparencies:measured at'] = current_time.strftime('%Y-%m-%dT%H:%M:%S')
new_sample['measured_on'] = current_time.strftime('%Y-%m-%d')
new_sample['activated_at'] = current_time.strftime('%Y-%m-%d %H:%M:%S')


# 1. Validate new sample

sample_errors = []

keys = ['organization_id', 'org_name', 'site_id', 'site_name', 'latitude', 'longitude', 'elevation', 'measured_on',
        'transparencies:userid', 'transparencies:measured at', 'transparencies:comments',
        'transparencies:water body state', 'transparencies:transparency disk image disappearance (m)',
        'transparencies:transparency disk does not disappear', 'transparencies:tube image disappearance (cm)',
        'transparencies:tube image does not disappear', 'transparencies:sensor turbidity ntu', 'transparencies:sensor mfg',
        'transparencies:sensor model', 'transparencies:globe teams']

# Check for missing keys
for key in [k for k in keys if k not in new_sample]:
    sample_errors.append(f"Missing keys in new_sample: {key}")

# Check for missing measurement
if pd.isna(new_sample['transparencies:transparency disk image disappearance (m)']) and \
        pd.isna(new_sample['transparencies:tube image disappearance (cm)']):
    sample_errors.append("At least one of disk or tube image disappearance must have a measurement.")

# Notify GLOBE administrator and user if there are validation errors
if sample_errors:
    sample_info = f"\n\nNew Sample:\n\n{new_sample}\n\nValidation Errors:\n\n" + '\n'.join(sample_errors)
    send_email(GLOBE_ADMIN_EMAIL,
               f"Validation Errors for New Sample",
               f"The following validation errors have been detected for the new sample. User has been notified.{sample_info}")
    send_email('globe_user@example.com',
               "Your sample has been rejected",
               f"Your sample has been rejected due to validation errors. GLOBE administrators have been notified.{sample_info}")


# 2. Validate water transparency measurement

# Assign unique sample ID
site_id = new_sample.get('site_id')
usid = int(str(site_id) + str(datetime.now().timestamp()).replace('.', ''))

# Set measurement device used
device = 'disk' if pd.notna(new_sample['transparencies:transparency disk image disappearance (m)']) else 'tube'

# Select and load current model
if site_id in TRUSTED_SITES:
    try:
        # Try loading existing BHM model
        with open('model_bhm.pkl', 'rb') as f:
            current_model = pickle.load(f)
    except Exception as e:
        # Train new model if no existing model is found
        print(f"Error loading 'model_bhm.pkl': {e}. Training new model…")
        current_model = bayesian_hierarchical_model(main_df)
else:
    if device == 'disk':
        try:
            # Try loading existing disk RF model
            with open('model_rf_disk.pkl', 'rb') as f:
                current_model = pickle.load(f)
        except Exception as e:
            # Train new model if no existing model is found
            print(f"Error loading 'model_rf_disk.pkl': {e}. Training new model…")
            current_model = random_forest_model_test(main_df, device)
    else:
        try:
            # Try loading existing tube RF model
            with open('model_rf_tube.pkl', 'rb') as f:
                current_model = pickle.load(f)
        except Exception as e:
            # Train new model if no existing model is found
            print(f"Error loading '_model_rf_tube.pkl': {e}. Training new model…")
            current_model = random_forest_model_test(main_df, device)

# Convert new_sample to dataframe
new_sample_df = pd.DataFrame(new_sample, index=[0])

# Compute measurement prediction for new sample
new_sample_X, new_sample_y = prepare_rf_data(new_sample_df, device)
prediction = current_model.predict(new_sample_X)[0]
difference = np.abs(prediction - new_sample_y.values[0])
print(f"[New sample ID: {usid}] \n Device: {device} \n Actual: {new_sample_y[0]} \n Prediction: {prediction} \n Difference: {difference}")

# Add prediction to new sample
if device == 'disk':
    new_sample_df['predictions:transparency disk image disappearance (m)'] = prediction
else:
    new_sample_df['predictions:tube image disappearance (cm)'] = prediction


# 3. Add new sample to main_df or pending_review_df dataframe
if difference > 5:
    # Notify user if prediction and measurement differ by more than 5 cm
    send_email('globe_user@example.com',
               'Your GLOBE sample',
               'Please check the measurements of your new GLOBE sample. They differ by 5 cm from the predicted value' + \
               f'Predicted value: {prediction}, Your measurement: {new_sample_y}')
    # Set status of new sample to hold
    new_sample_df['sample_status'] = 'hold'
    print(f"New sample, {usid} is on hold due to measurement discrepancy. User has been notified.")
    # Add new sample to pending review dataframe
    pending_review_df = pd.concat([pending_review_df, new_sample_df], ignore_index=True)
    # Exit execution
    # exit()
else:
    # Notify user
    send_email('globe_user@example.com',
               'Your GLOBE sample',
               'Your GLOBE sample has been successfully submitted. Thank you for your contribution!')
    # Set status of new sample to hold
    new_sample_df['sample_status'] = 'active'
    # Add new sample to main dataframe
    main_df = pd.concat([main_df, new_sample_df], ignore_index=True)


# 4. Update model

# Create filtered dataframe for remodeling
if device == 'disk':
    modeling_df = main_df[(np.abs(main_df['predictions:transparency disk image disappearance (m)'] - main_df['transparencies:transparency disk image disappearance (m)']) < 5) | (main_df['predictions:transparency disk image disappearance (m)'].isna())].copy()
else:
    modeling_df = main_df[(np.abs(main_df['predictions:tube image disappearance (cm)'] - main_df['transparencies:tube image disappearance (cm)']) < 5) |
                          (main_df['predictions:tube image disappearance (cm)'].isna())].copy()

# Calculate modeling weight based on sample age
current_date = datetime.now()
modeling_df['measured_on'] = pd.to_datetime(modeling_df['measured_on'])
modeling_df['sample_age_years'] = (current_date - modeling_df['measured_on']).dt.days / 365.25
modeling_df['weight'] = modeling_df['sample_age_years'].apply(lambda age: max(0, 1 - age / 10))

# Create new model
if type(current_model) == RandomForestRegressor:
    new_model = random_forest_model_test(modeling_df, device, apply_weight=True)
else:
    new_model = bayesian_hierarchical_model(modeling_df)


# 5. Test new model

model_errors = []

# Test object type
if type(current_model) != type(new_model):
    model_errors.append("New model is not of the same type as the current model.")

# Test accuracy

# Notify GLOBE administrator and user if there are validation errors
if model_errors:
    model_info = f"\n\nNew Model:\n\n{type(new_model)}\n\nValidation Errors:\n\n" + '\n'.join(model_errors)
    send_email(GLOBE_ADMIN_EMAIL,
               f"Validation Errors for New Model",
               f"The following validation errors have been detected for the new model. {model_info}")
    # Exit execution
    # exit()


# 6. Deploy new model

# Set model file name
if type(new_model) == RandomForestRegressor:
    if device == 'disk':
        model_name = 'model_rf_disk.pkl'
    else:
        model_name = 'model_rf_tube.pkl'
else:
    model_name = 'model_bhm.pkl'

# Save model file
with open(model_name, 'wb') as f:
    pickle.dump(new_model, f)
