import { useEffect, useRef, useState } from 'react'
import { Popover } from 'react-tiny-popover'
import MenuItem from './MenuItem'

interface IMenuItemDropdown {
  label: string
  icon: string
  typeName: string
  type: any
}

const MenuItemDropdown = ({ label, icon, typeName, type }: IMenuItemDropdown) => {
  const [isPopoverOpen, setIsPopoverOpen] = useState(false)
  const popoverContainerRef = useRef<HTMLDivElement>(null)

  const toggleIsPopoverOpen = () => {
    if (!isPopoverOpen) {
      setIsPopoverOpen(true)
      setTimeout(() => {
        document.body.addEventListener('click', bodyClickEventLister)
      }, 200)
    }
  }

  const bodyClickEventLister = () => {
    document.body.removeEventListener('click', bodyClickEventLister)
    popoverContainerRef?.current?.classList?.add('closing')

    setTimeout(() => {
      popoverContainerRef?.current?.classList?.remove('closing')
      setIsPopoverOpen(false)
    }, 1)
  }

  useEffect(() => {
    const popoverElement = document.getElementsByClassName(
      typeName + '-popover'
    )[0] as HTMLElement
    const parentElement = document.getElementsByClassName(
      typeName + '-parent'
    )[0] as HTMLElement

    if (popoverElement) {
      popoverElement.style.width = parentElement.offsetWidth + 'px'
    }
  }, [isPopoverOpen, type, typeName])

  return (
    <Popover
      ref={popoverContainerRef}
      parentElement={
        document.getElementsByClassName(typeName + '-parent')[0] as HTMLElement
      }
      containerClassName={`${typeName}-popover popover-container`}
      isOpen={isPopoverOpen}
      content={
        <div className="popover-content">
          {Object.values(type).map((item) => {
            return <div key={item as string}>{item as string}</div>
          })}
        </div>
      }>
      <>
        <MenuItem
          className={`${typeName}-parent`}
          label={label}
          icon={icon}
          onClick={toggleIsPopoverOpen}
        />
      </>
    </Popover>
  )
}

export default MenuItemDropdown
