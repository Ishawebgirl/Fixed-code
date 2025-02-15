import { useEffect, useRef, useState } from 'react'
import Icon from './Icon'

interface IFilterTableMenuItem {
  filterTerms: string
  onChange: (terms: string) => void
}

const FilterTableMenuItem = ({ filterTerms, onChange }: IFilterTableMenuItem) => {
  const [isOpen, setIsOpen] = useState(false)
  const [terms, setTerms] = useState('')

  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setTerms(filterTerms)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    onChange(terms)
  }, [terms]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (isOpen && inputRef?.current) {
      inputRef.current.focus()
    } else {
      setTerms('')
    }
  }, [isOpen])

  return (
    <>
      <div onClick={() => setIsOpen(!isOpen)}>
        <Icon name="MagnifyingGlass"></Icon>Filter
        <div
          onClick={(e) => e.stopPropagation()}
          className={`filter-input ${isOpen ? 'open' : ''}`}>
          <input
            ref={inputRef}
            onChange={(e) => setTerms(e.currentTarget.value)}
            value={terms}></input>
        </div>
      </div>
    </>
  )
}

export default FilterTableMenuItem
