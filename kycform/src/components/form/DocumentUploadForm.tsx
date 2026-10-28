import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { useKYC } from '../../context/KYCContext';
import { FormField } from '../shared/FormField';
import { FileUpload } from '../shared/FileUpload';
import type { DocumentUpload, DocumentType } from '../../types/kyc';

const DOCUMENT_TYPES: { value: DocumentType; label: string; hasBack: boolean; hasExpiry: boolean }[] = [
  { value: 'pan', label: 'PAN Card', hasBack: false, hasExpiry: false },
  { value: 'passport', label: 'Passport', hasBack: false, hasExpiry: true },
  { value: 'driving_licence', label: 'Driving Licence', hasBack: true, hasExpiry: true },
  { value: 'voter_id', label: 'Voter ID', hasBack: true, hasExpiry: false },
  { value: 'aadhaar', label: 'Aadhaar Card', hasBack: true, hasExpiry: false },
  { value: 'other', label: 'Other Identity Document', hasBack: true, hasExpiry: true },
];

interface DocumentFormProps {
  onNext: () => void;
  onBack: () => void;
}

export function DocumentUploadForm({ onNext, onBack }: DocumentFormProps) {
  const { application, updateDocument } = useKYC();
  const [frontError, setFrontError] = useState('');
  const [backError, setBackError] = useState('');

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
    reset,
  } = useForm<DocumentUpload>({
    defaultValues: application.document,
  });

  useEffect(() => {
    reset(application.document);
  }, []);

  const selectedType = watch('documentType') as DocumentType | '';
  const docConfig = DOCUMENT_TYPES.find((d) => d.value === selectedType);

  const validate = (_data: DocumentUpload): boolean => {
    let valid = true;
    if (!application.document.frontImagePreview) {
      setFrontError('Front image is required.');
      valid = false;
    } else {
      setFrontError('');
    }
    if (docConfig?.hasBack && !application.document.backImagePreview) {
      setBackError('Back image is required for this document type.');
      valid = false;
    } else {
      setBackError('');
    }
    return valid;
  };

  const onSubmit = (data: DocumentUpload) => {
    if (!validate(data)) return;
    updateDocument(data);
    onNext();
  };

  return (
    <form className="step-form" onSubmit={handleSubmit(onSubmit)} noValidate>
      <div className="form-section-header">
        <h2>Identity Document Upload</h2>
        <p>
          Please upload a valid government-issued identity document. Use fictional or sample
          images during development and demonstrations.
        </p>
      </div>

      <div className="form-grid">
        <FormField
          label="Document Type"
          required
          error={errors.documentType?.message}
          htmlFor="documentType"
        >
          <select
            id="documentType"
            className={`form-input ${errors.documentType ? 'input-error' : ''}`}
            {...register('documentType', { required: 'Please select a document type.' })}
            onChange={(e) => {
              updateDocument({
                documentType: e.target.value as DocumentType,
                frontImageRef: undefined,
                backImageRef: undefined,
                frontImagePreview: undefined,
                backImagePreview: undefined,
              });
            }}
          >
            <option value="">— Select document type —</option>
            {DOCUMENT_TYPES.map((d) => (
              <option key={d.value} value={d.value}>
                {d.label}
              </option>
            ))}
          </select>
        </FormField>

        <FormField
          label="Document Number"
          required
          error={errors.documentNumber?.message}
          htmlFor="documentNumber"
        >
          <input
            id="documentNumber"
            type="text"
            className={`form-input ${errors.documentNumber ? 'input-error' : ''}`}
            placeholder="Enter document number (use fictional data)"
            {...register('documentNumber', {
              required: 'Document number is required.',
            })}
          />
        </FormField>

        <FormField label="Issuing Authority / Country" htmlFor="issuingAuthority">
          <input
            id="issuingAuthority"
            type="text"
            className="form-input"
            placeholder="e.g. Government of India"
            {...register('issuingAuthority')}
          />
        </FormField>

        <FormField label="Issue Date" htmlFor="issueDate">
          <input
            id="issueDate"
            type="date"
            className="form-input"
            {...register('issueDate')}
          />
        </FormField>

        {docConfig?.hasExpiry && (
          <FormField label="Expiry Date" htmlFor="expiryDate">
            <input
              id="expiryDate"
              type="date"
              className="form-input"
              {...register('expiryDate')}
            />
          </FormField>
        )}
      </div>

      {selectedType && (
        <div className="upload-section">
          <h3 className="subsection-title">Document Images</h3>
          <p className="upload-disclaimer">
            ⚠️ Use fictional or sample images for this demonstration. Do not upload real identity
            documents.
          </p>
          <div className="upload-grid">
            <FileUpload
              label="Document Image"
              side="front"
              preview={application.document.frontImagePreview}
              required
              error={frontError}
              onFileSelected={(file, dataUrl) => {
                setFrontError('');
                updateDocument({
                  frontImageRef: file.name,
                  frontImagePreview: dataUrl,
                });
              }}
              onRemove={() =>
                updateDocument({ frontImageRef: undefined, frontImagePreview: undefined })
              }
            />

            {docConfig?.hasBack && (
              <FileUpload
                label="Document Image"
                side="back"
                preview={application.document.backImagePreview}
                required
                error={backError}
                onFileSelected={(file, dataUrl) => {
                  setBackError('');
                  updateDocument({
                    backImageRef: file.name,
                    backImagePreview: dataUrl,
                  });
                }}
                onRemove={() =>
                  updateDocument({ backImageRef: undefined, backImagePreview: undefined })
                }
              />
            )}
          </div>
        </div>
      )}

      <div className="form-actions">
        <button type="button" className="btn btn-secondary" onClick={onBack}>
          Back
        </button>
        <button type="submit" className="btn btn-primary">
          Continue
        </button>
      </div>
    </form>
  );
}
