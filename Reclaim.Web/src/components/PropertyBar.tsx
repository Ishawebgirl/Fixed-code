import { useEffect, useState } from 'react'
import Icon from './Icon'
import HTMLReactParser from 'html-react-parser'

interface IPropertyBar {
  children: any
  entityID: string | null
  isVisible: boolean
  onSave: any
  onCancel: any
}

const PropertyBar = ({
  children,
  entityID,
  isVisible,
  onSave,
  onCancel,
}: IPropertyBar) => {
  const [errorMessage, setErrorMessage] = useState('')

  const handleSubmit = async (): Promise<boolean> => {
    if (!validate()) {
      return false
    }

    await onSave()
      .then((result: any) => {
        onCancel()
        flashRow(entityID!)
        return true
      })
      .catch((error: any) => {
        if (error.response) {
          const apiError = JSON.parse(error.response)

          if (apiError.message) {
            setErrorMessage(apiError.message)
          } else {
            setErrorMessage(error.response)
          }
        }

        return false
      })

    return false
  }

  const validate = (): boolean => {
    const invalidInputs = document.getElementsByClassName('not-valid')

    if (invalidInputs.length > 0) {
      setErrorMessage('Please complete all the required fields.')
      return false
    } else {
      setErrorMessage('')
      return true
    }
  }

  const flashRow = (id: string) => {
    const row = document.getElementById(`row-${id}`)

    if (row) {
      row.classList.add('flash-row')
      setTimeout(() => {
        row.classList.remove('flash-row')
      }, 750)
    }
  }

  useEffect(() => {
    setErrorMessage('')
  }, [entityID])

  return (
    <nav id="property-bar" className={`property-bar${isVisible ? '' : ' collapsed'}`}>
      <form onSubmit={handleSubmit}>
        <div className="collapse-button" onClick={onCancel}>
          <Icon name="AngleDoubleRight"></Icon>
        </div>
        {children}
        <div className="row buttons">
          <div className="col-6">
            <button
              disabled={false}
              type="button"
              onClick={handleSubmit}
              className="styled-button">
              Save
            </button>
          </div>
          <div className="col-6">
            <button
              disabled={false}
              type="button"
              onClick={onCancel}
              className="styled-button cancel">
              Cancel
            </button>
          </div>
        </div>
        <div className="error-message">{HTMLReactParser(errorMessage)}</div>
      </form>
    </nav>
  )
}

export default PropertyBar
