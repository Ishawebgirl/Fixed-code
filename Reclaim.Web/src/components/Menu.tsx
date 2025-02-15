import { ReactElement } from 'react'
import MenuItem from './MenuItem'
import React from 'react'

interface IMenu {
  children: ReactElement<typeof MenuItem> | Array<ReactElement<typeof MenuItem>>
}

const Menu = ({ children }: IMenu) => {
  return <div className="menu">{children}</div>
}

export default Menu
