import { States } from 'helpers/states'
import moment from 'moment'
import { useEffect, useState } from 'react'
import InputMask from 'react-input-mask'

interface ITextInput {
  entityID?: string
  label: string
  name: string
  type: string
  required?: boolean
  fixed?: boolean
  value: string
  group?: string
  groupError?: string
  regex?: RegExp
  columnSpec?: string
  onChange: any
  formatting?: string
}

const TextInput = ({
  entityID = '',
  type = 'text',
  required = false,
  fixed = false,
  label,
  name,
  value,
  group,
  groupError,
  regex,
  columnSpec,
  onChange,
  formatting,
}: ITextInput) => {
  const [text, setText] = useState('')
  const [isValid, setIsValid] = useState(true)
  const [isGroupValid, setIsGroupValid] = useState(true)

  let labelEx = label

  if (fixed) {
    labelEx += ' (fixed)'
  } else if (!required) {
    labelEx += ' (optional)'
  }

  useEffect(() => {
    if (!value || value.length === 0) {
      setText('')
      return
    }

    switch (type) {
      case 'date':
        try {
          const formatted = moment.utc(value).format('MM/DD/YYYY')
          setText(formatted)
          setIsValid(true)
        } catch {}
        break

      default:
        setText(value)
        setIsValid(true)
        break
    }
  }, [entityID, type, value])

  useEffect(() => {
    if (group != null && groupError === group && text.length === 0) {
      setIsGroupValid(false)
    } else {
      setIsGroupValid(true)
    }
  }, [group, text, groupError])

  const handleChange = (e: any) => {
    let value = e.target.value

    switch (formatting) {
      case 'uppercase':
        value = value.toUpperCase()
        break
    }

    setText(value)

    if (!regex) {
      setIsValid(!required || value.length > 0)
    }

    if (typeof onChange === 'function') {
      onChange(value)
    }
  }

  const handleBlur = (e: any) => {
    if (regex) {
      const val = e.target.value.trim()
      setText(val)

      if (val.length === 0 && !required) {
        setIsValid(true)
      } else {
        setIsValid(new RegExp(regex).test(val))
      }
    }
  }

  const handleChangeDate = (e: any) => {
    if (e.target.value.length === 0) {
      return
    }
    setText(e.target.value)

    if (typeof onChange === 'function') {
      onChange(e.target.value)
    }
  }

  const handleBlurDate = (e: any) => {
    if (e.target.value.length === 0) {
      return
    }

    handleBlur(e)
  }

  const className = `input-container ${isValid ? 'valid' : 'not-valid'}${isGroupValid ? '' : ' group-not-valid'} type-${type} ${columnSpec == null ? '' : 'column-' + columnSpec}`
  const groupName = group != null ? `group-${group}` : ''

  switch (type) {
    case 'state':
      return (
        <div className={className}>
          <select
            className={groupName}
            required={required!}
            name={name}
            value={text}
            onChange={handleChangeDate}
            onBlur={handleBlurDate}>
            <option value=" " className="empty-option"></option>
            {States.map((x, y) => (
              <option key={y} value={x.abbreviation}>
                {x.name}
              </option>
            ))}
          </select>
          <label className={text && 'filled'} htmlFor={name}>
            {labelEx}
          </label>
        </div>
      )

    case 'telephone':
      return (
        <div className={className}>
          <InputMask
            className={groupName}
            required={required!}
            readOnly={fixed}
            name={name}
            mask="+1 999-999-9999"
            value={text}
            onChange={handleChangeDate}
            onBlur={handleBlurDate}
          />
          <label className={text && 'filled'} htmlFor={name}>
            {labelEx}
          </label>
        </div>
      )

    case 'date':
      return (
        <div className={className}>
          <InputMask
            className={groupName}
            required={required!}
            readOnly={fixed}
            name={name}
            placeholder="MM/DD/YYYY"
            mask="99/99/9999"
            value={text}
            onChange={handleChangeDate}
            onBlur={handleBlurDate}
          />
          <label className="filled" htmlFor={name}>
            {labelEx}
          </label>
        </div>
      )

    case 'dateTime':
      return (
        <div className={className}>
          <input
            className={groupName}
            required={required!}
            readOnly={fixed}
            name={name}
            value={
              text == null ? '' : moment(text).format('MM/DD/YYYY hh:mmA').toLowerCase()
            }
            onChange={handleChangeDate}
            onBlur={handleBlurDate}
          />
          <label className="filled" htmlFor={name}>
            {labelEx}
          </label>
        </div>
      )

    case 'textArea':
      return (
        <div className={className}>
          <textarea
            className={groupName}
            required={required!}
            readOnly={fixed}
            name={name}
            value={text}
            onChange={handleChange}
            onBlur={handleBlur}
          />
          <label className={text && 'filled'} htmlFor={name}>
            {labelEx}
          </label>
        </div>
      )

    default:
      return (
        <div className={className}>
          <input
            className={groupName}
            required={required!}
            readOnly={fixed}
            name={name}
            type={type}
            value={text}
            onChange={handleChange}
            onBlur={handleBlur}
          />
          <label className={text && 'filled'} htmlFor={name}>
            {labelEx}
          </label>
        </div>
      )
  }
}

export default TextInput
